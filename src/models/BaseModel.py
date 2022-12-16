# -*- coding: UTF-8 -*-

from matplotlib.pyplot import phase_spectrum
from numpy.testing._private.utils import print_assert_equal
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import NoReturn, List

from utils import utils
from helpers.BaseReader import BaseReader


class BaseModel(nn.Module):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--model_path', type=str, default='',
                            help='Model save path.')
        parser.add_argument('--buffer', type=int, default=1,
                            help='Whether to buffer feed dicts for dev/test')
        return parser

    @staticmethod
    def init_weights(m):
        
        if 'Linear' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)


    def __init__(self, args, corpus: BaseReader):
        super(BaseModel, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
        self.model_path = args.model_path
        self.buffer = args.buffer
        self.optimizer = None
        self.check_list = list()  # observe tensors in check_list every check_epoch

        self._define_params()
        self.total_parameters = self.count_variables()
        logging.info('#params: %d' % self.total_parameters)

    """
    Key Methods
    """
    def _define_params(self) -> NoReturn:
        pass

    def forward(self, feed_dict: dict) -> dict:
        """
        :param feed_dict: batch prepared in Dataset
        :return: out_dict, including prediction with shape [batch_size, n_candidates]
        """
        pass

    def loss(self, out_dict: dict) -> torch.Tensor:
        pass
    def inference(self,data):
        return self.forward(data)

    """
    Auxiliary Methods
    """
    def customize_parameters(self) -> list:
        # customize optimizer settings for different parameters
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        return optimize_dict

    def save_model(self, model_path=None) -> NoReturn:
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(model_path)
        torch.save(self.state_dict(), model_path)
        # logging.info('Save model to ' + model_path[:50] + '...')

    def load_model(self, model_path=None) -> NoReturn:
        if model_path is None:
            model_path = self.model_path
        print(model_path)
        self.load_state_dict(torch.load(model_path))
        logging.info('Load model from ' + model_path)

    def count_variables(self) -> int:
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def actions_before_train(self):  # e.g., re-initial some special parameters
        pass

    def actions_after_train(self):  # e.g., save selected parameters
        pass

    """
    Define Dataset Class
    """
    class Dataset(BaseDataset):
        def __init__(self, model, corpus, phase: str):
            self.model = model  # model object reference
            self.corpus = corpus  # reader object reference
            self.phase = phase  # train / dev / test

            self.buffer_dict = dict()
            self.buffer = self.model.buffer and 'train' not in self.phase
            print(phase,corpus.data_df.keys())
            self.data = utils.df_to_dict(corpus.data_df[phase])
            # ↑ DataFrame is not compatible with multi-thread operations

            self._prepare()

        def __len__(self):
            if type(self.data) == dict:
                for key in self.data:
                    return len(self.data[key])
            return len(self.data)

        def __getitem__(self, index: int) -> dict:
            return self.buffer_dict[index] if self.buffer else self._get_feed_dict(index)

        # Prepare model-specific variables and buffer feed dicts
        def _prepare(self) -> NoReturn:
            if self.buffer:
                for i in tqdm(range(len(self)), leave=False, desc=('Prepare ' + self.phase)):
                    self.buffer_dict[i] = self._get_feed_dict(i)

        # ! Key method to construct input data for a single instance
        def _get_feed_dict(self, index: int) -> dict:
            pass

        # Called before each training epoch
        def actions_before_epoch(self) -> NoReturn:
            pass

        # Collate a batch according to the list of feed dicts
        def collate_batch(self, feed_dicts: List[dict]) -> dict:
            np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
            feed_dict = dict()
            for key in feed_dicts[0]:
                stack_val = np.array([d[key] for d in feed_dicts])
                if stack_val.dtype == np.object:  # inconsistent length (e.g., history)
                  
                    feed_dict[key] = pad_sequence([torch.from_numpy(x) for x in stack_val], batch_first=True)
                else:
                    feed_dict[key] = torch.from_numpy(stack_val)
            feed_dict['batch_size'] = len(feed_dicts)
            feed_dict['phase'] = self.phase
            for k in feed_dict.keys():
                if 'history_items' in k and feed_dict[k].dtype!=torch.int64:
                    feed_dict[k]=feed_dict[k].long()
            return feed_dict


class GeneralModel(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--num_neg', type=int, default=1,
                            help='The number of negative items during training.')
        parser.add_argument('--dropout', type=float, default=0,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--test_all', type=int, default=0,
                            help='Whether testing on all the items.')
        return BaseModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items
        self.num_neg = args.num_neg
        self.dropout = args.dropout
        self.test_all = args.test_all
        super().__init__(args, corpus)

    def loss(self, out_dict: dict) -> torch.Tensor:
        """
        BPR ranking loss with optimization on multiple negative samples (a little different now)
        "Recurrent neural networks with top-k gains for session-based recommendations"
        :param out_dict: contain prediction with [batch_size, -1], the first column for positive, the rest for negative
        :return:
        """
        predictions = out_dict['prediction']
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        loss = -(((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).clamp(min=1e-5,max=100).log()).mean()
        # neg_pred = (neg_pred * neg_softmax).sum(dim=1)
        # loss = F.softplus(-(pos_pred - neg_pred)).mean()
        # ↑ For numerical stability, we use 'softplus(-x)' instead of '-log_sigmoid(x)'
        return loss

    class Dataset(BaseModel.Dataset):
        def _get_feed_dict(self, index):
            user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
            if 'train' not in self.phase and self.model.test_all:
                
                neg_items = np.arange(1, self.corpus.n_items)
            else:
                neg_items = self.data['neg_items'][index]
            item_ids = np.concatenate([[target_item], neg_items]).astype(int)
            feed_dict = {
                'user_id': user_id,
                'item_id': item_ids
            }
            return feed_dict

        # Sample negative items for all the instances
        def actions_before_epoch(self) -> NoReturn:
            
            neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.model.num_neg))
            for i, u in enumerate(self.data['user_id']):
                clicked_set = self.corpus.train_clicked_set[u]  # neg items are possible to appear in dev/test set
                for j in range(self.model.num_neg):
                    while neg_items[i][j] in clicked_set:
                        neg_items[i][j] = np.random.randint(1, self.corpus.n_items)
            self.data['neg_items'] = neg_items


class SequentialModel(GeneralModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--history_max', type=int, default=20,
                            help='Maximum length of history.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.history_max = args.history_max
        super().__init__(args, corpus)

    class Dataset(GeneralModel.Dataset):
        def _prepare(self):
            idx_select = np.array(self.data['position']) > 0  # history length must be non-zero
            for key in self.data:
                self.data[key] = np.array(self.data[key])[idx_select]
            super()._prepare()

        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            
            pos = self.data['position'][index]
            user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
            if self.model.history_max > 0:
                user_seq = user_seq[-self.model.history_max:]
            
            

            feed_dict['history_items'] = np.array([x[0] for x in user_seq])
            feed_dict['history_times'] = np.array([x[1] for x in user_seq])
            feed_dict['lengths'] = len(feed_dict['history_items'])
            feed_dict['position']=pos
            return feed_dict



class MultiModel(GeneralModel):
    """
    Here our MultiModel is sequence and graph model it may be extended for feature
    
    """

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--history_max', type=int, default=20,
                            help='Maximum length of history.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.history_max = args.history_max
        self.behaviors=corpus.behaviors
        super().__init__(args, corpus)
        


    class Dataset(GeneralModel.Dataset):


        def __init__(self, model, corpus, phase: str):
           
            super().__init__(model, corpus, phase)
            

        def make_ti2dfi(self,data,target_behavior):
            index=list(range(len(data['user_id'])))
            ti2dfi=dict()
            
            for i,b in zip(index,data['behavior']):
                
                if b==target_behavior:
                    ti2dfi[len(ti2dfi)]=i
                    # print(b,ti2dfi[len(ti2dfi)-1])
            return ti2dfi
        def __len__(self):
            return len(self.ti2dfi)
            
        def _prepare(self):
            index_select = np.array(self.data['position']) == 0  # history length must be non-zero
            user_ids=list(set(self.data['user_id']))
            pass_id_set=list()
            
            for u in user_ids:
                
                target_user_seq=self.corpus.user_his[u][self.corpus.target_behavior]
                if len(target_user_seq)==0:
                    continue
                aux_p=target_user_seq[0][2]
                if sum(aux_p)==0:
                    pass_id_set.append(u)

            is_pass_id=np.isin(np.array(self.data['user_id']),pass_id_set)
            index_select=is_pass_id&index_select
            index_select=~index_select
            for key in self.data:
                self.data[key] = np.array(self.data[key])[index_select]
            self.ti2dfi=self.make_ti2dfi(self.data,self.corpus.target_behavior) # this is the target index 2 real index in df, note that the real df include train dev test    
            super()._prepare()

        def _get_feed_dict(self, index):
            
            df_index=self.ti2dfi[index]
            feed_dict = super()._get_feed_dict(df_index)
            pos = self.data['position'][df_index]
            user_seq = self.corpus.user_his[feed_dict['user_id']]
            user_tag_seq=user_seq[self.corpus.target_behavior][:pos]
            aux_poses=user_seq[self.corpus.target_behavior][pos][2]
            aux_seqs=[ user_seq[b][:p] for b,p in zip(self.corpus.aux_behaviors,aux_poses)]
            if self.model.history_max > 0:
                user_tag_seq = user_tag_seq[-self.model.history_max:]
                aux_seqs=[aux[-self.model.history_max:] for aux in aux_seqs]
            feed_dict['%s_history_items' %self.corpus.target_behavior]=np.array([x[0] for x in user_tag_seq])
            if len(feed_dict['%s_history_items' %self.corpus.target_behavior])>0:
                feed_dict['%s_have_behavior' %self.corpus.target_behavior]=1
            else:
                feed_dict['%s_have_behavior' %self.corpus.target_behavior]=0
            feed_dict['%s_lengths'%self.corpus.target_behavior] = len(feed_dict['%s_history_items'%self.corpus.target_behavior])
            for aux_seq,b in zip(aux_seqs,self.corpus.aux_behaviors):
                if len(aux_seq)==0:
                    feed_dict['%s_have_behavior'%b]=0
                else:
                    feed_dict['%s_have_behavior'%b]=1
                feed_dict['%s_history_items' %b]=np.array([x[0] for x in aux_seq])
                feed_dict['%s_lengths'%b] = len(feed_dict['%s_history_items'%b])
            
            # next_aux_poses=next_target_b[]
            return feed_dict

        def actions_before_epoch(self) -> NoReturn:
            
            neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self.data['user_id']), self.model.num_neg))
            index=list(range(len(self.data['user_id'])))
            for i, u,b in zip(index,self.data['user_id'],self.data['behavior']):
                
                clicked_set = self.corpus.train_clicked_set[u]  # neg items are possible to appear in dev/test set
                for j in range(self.model.num_neg):
                    if b!=self.corpus.target_behavior:
                        neg_items[i][j]=0
                        continue

                    while neg_items[i][j] in clicked_set:
                        neg_items[i][j] = np.random.randint(1, self.corpus.n_items)
            self.data['neg_items'] = neg_items