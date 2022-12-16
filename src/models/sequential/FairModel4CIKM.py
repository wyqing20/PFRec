# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" SASRec
Reference:
    "Self-attentive Sequential Recommendation"
    Kang et al., IEEE'2018.
Note:
    When incorporating position embedding, we make the position index start from the most recent interaction.
CMD example:
    for the pretrian:
    python main.py --model_name Pmodel --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 \
    --history_max 20 --dataset 'CIKM10%'

    for the fine-tuning:
         python main_fair.py --model_name FairModel4CIKM --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 100 --dataset 'CIKM10%' --epoch 100 --epoch_step 3000 --stage 5
"""



from concurrent.futures.process import _chain_from_iterable_of_lists
from sqlite3 import adapters
from typing import List
from networkx.algorithms.operators.product import power
from numpy.core.defchararray import not_equal
from numpy.core.fromnumeric import mean
from numpy.core.records import record
from numpy.testing._private.utils import print_assert_equal
import torch
from torch._C import default_generator
from torch.cuda import random
import torch.nn as nn
import numpy as np
from torch.nn.modules.linear import Linear
from models.BaseModel import SequentialModel
from utils import layers
import logging
import os
from tqdm import tqdm
from utils import utils
import random
import torch.nn.functional as F

class FairModel4CIKM(SequentialModel):
    reader='PReaderFair'
    runner='FairRunnerCIKM'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads','f_encoder','stage','history_max','encoder','profile','d_step','l_weight','tag']
   
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--stage', type=int, default=1,
                            help='Stage of training: 1-pretrain, 2-pmodel,3-fine-tuning,4:eval default-from_scratch. 10-pretrain prompt on warm-set \
                                11 train on the prompt set 12: I forget it  13 fine-tuning and not change item-embedding\
                                    14 fine-tuing & change item embedding 15 p-tuning zero-shot 16 prompt-zero-shot')
        parser.add_argument('--encoder', type=str, default='SASRec',
                            help='Choose a sequence encoder: GRU4Rec, Caser, BERT4Rec.')
        parser.add_argument('--hidden_size',type=int,default=64)
        parser.add_argument('--autoint_layers', type=int, default=1,
                            help='Number of autoInt self-attention layers.')
        parser.add_argument('--autoint_heads', type=int, default=1,
                            help='Number of autoInt heads.')
        parser.add_argument('--f_encoder', type=str, default='Linear',
                            help='Number of autoInt heads.')
        parser.add_argument("--w_feature",type=int,default=0,help='pre train with side inf')
        parser.add_argument("--profile",type=str,default='old',help='predict profile')
        parser.add_argument('--tag',type=str,default='',help='some desciribe of model')
        
        
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.encoder_name = args.encoder
        self.year_num=corpus.n_years
        self.power_num=corpus.n_powers
        self.stage=args.stage
        self.hidden_size=args.hidden_size
        self.autoInt_layers=args.autoint_layers
        self.autoInt_heads=args.autoint_heads
        self.f_encoder_name=args.f_encoder
        self.w_feature=args.w_feature
        self.predict_profile=args.profile
        self.d_optimizer=None
        self.g_optimizer=None
        super().__init__(args, corpus)
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self.pre_path = '/data/wuyq/ReChorus/model/FairModel4CIKM/Pre__{}__encoder={}__w_feature={}__max_history={}__num_layers{}__num_heads{}.pt'.format(corpus.dataset, self.encoder_name,self.w_feature,
                    self.max_his,self.num_layers,self.num_heads)
        if self.stage==1:
            self.model_path = self.pre_path 
        
        # self.data=utils.df_to_dict(corpus.data_df['train1'])  
        
        # self.mask_token=corpus.mask_token

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(model_path)
        torch.save({'state_dict':self.state_dict(),'g_optimizer':self.g_optimizer,'d_optimizer':self.d_optimizer}, model_path)
    def load_model(self, model_path=None) :
        if model_path is None:
            model_path = self.model_path
        print(model_path)
        try:
            self.load_state_dict(torch.load(model_path)['state_dict'])
            logging.info('Load model from ' + model_path)
        except:
            self.load_state_dict(torch.load(model_path)) #for only save model  state dict
            logging.info('Load model from ' + model_path)

    def actions_before_train(self):
        
        if self.stage != 1:  # fine-tune
            print('pretrain path: ', self.pre_path)
            #这里很多stage我也忘了 大概有用的就是1和2和5
            if (self.stage==6 or self.stage==7  or self.stage==3 or self.stage==2 or self.stage==4 or self.stage==5 or self.stage==8 or self.stage==10 or self.stage==11 or self.stage==12 or self.stage==13 or self.stage==14 or self.stage==15 or self.stage==16)  :
                if os.path.exists(self.pre_path):
                    
                    self.load_model(self.pre_path)
                    
                else:
                    logging.info('Train from scratch!')
            else:
                logging.info('Train from scratch!')
            self.add_Param()
            logging.info(self)     
    

    def init_add_weights(self,m):
        logging.info(m)
        logging.info('add_init')
        if 'Linear' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def d_parameters(self):
        paramters=[]
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if  'prediction' in name:
                paramters.append(p)
                logging.info('descri:'+name)
        return  [{'params':paramters}]
    def g_parameters(self):
        paramters=[]
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            # if  'prediction' not  in name and 'i_embedding' not in name:
            if self.stage==2:
                if  'prefix'  in name or 'adapter'  in name or 'layer_norm' in name:
                    paramters.append(p)
                    logging.info('gend:'+name)
            if self.stage==5:
                 if  'fillter'  in name :
                    paramters.append(p)
                    logging.info('gend:'+name)
        return  [{'params':paramters}]     
    def all_parameters(self):
        paramters=[]
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if  'prediction' not  in name:
           
                paramters.append(p)
                logging.info('all:'+name)
        return  [{'params':paramters}]       

    def P_tuning(self):
        return self.customize_parameters()
                

    def add_Param(self):
        self.prefix_embedding=nn.Embedding(100,self.emb_size,padding_idx=0)
        self.prefix_q_embedding=nn.Embedding(100,self.emb_size,padding_idx=0)
        self.embed_dim=self.emb_size
        self.neg_slope=0.2
        if True:
            self.age_embedding=nn.Embedding(5+1,self.emb_size)
            self.gender_embedding=nn.Embedding(2+1,self.emb_size)
            self.occupation_embedding=nn.Embedding(9+1,self.emb_size)
            self.age_embedding.apply(self.init_embedding)
            self.gender_embedding.apply(self.init_embedding)
            self.occupation_embedding.apply(self.init_embedding)
        if self.predict_profile=='gender':
            self.prediction=torch.nn.Sequential(nn.Linear(self.emb_size,self.emb_size),
                            nn.ReLU(),
                            nn.Linear(self.emb_size,2)
            )
            self.feature_num=2
        if self.predict_profile=='old':
            self.prediction=torch.nn.Sequential(nn.Linear(self.emb_size,self.emb_size),
                            nn.ReLU(),
                            
                            nn.Linear(self.emb_size,5)
            )
            self.feature_num=5
        if self.predict_profile=='power':
            self.prediction=torch.nn.Sequential(nn.Linear(self.emb_size,self.emb_size),
                            nn.ReLU(),
                            nn.Linear(self.emb_size,9)
            )
            self.feature_num=9
       
        if self.predict_profile=='old':
            num=[3925.0, 23095, 19974,  9342,  2886,]
        if self.predict_profile=='gender':
            num=[20256,39646]
        if self.predict_profile=='power':
            num=[ 4233.0,  4702,  4677,  4401,  4066, 17001, 10504,  6277,  4041]
        weight=(1/torch.tensor(num))
        weight=weight/weight.sum()
        if self.feature_num==2:
            self.c_loss_fun=nn.BCELoss()
            self.classify_loss_fun=nn.BCELoss()
        if self.feature_num>2:
            self.c_loss_fun=nn.NLLLoss(weight)
            self.classify_loss_fun=nn.NLLLoss(weight)
        if self.stage==5:
            self.fillter=torch.nn.Sequential(nn.Linear(self.emb_size,self.emb_size),
                                nn.ReLU(),
                                nn.Linear(self.emb_size,self.emb_size),
                                
                )
            self.fillter.apply(self.init_add_weights)
        self.adapter1=torch.nn.Sequential( nn.LayerNorm(self.emb_size),
                                nn.Linear(self.emb_size,self.emb_size//4),
                                nn.ReLU(),
                                nn.Linear(self.emb_size//4,self.emb_size),
                               
                                # nn.LayerNorm(self.emb_size)
                )
        self.adapter2=torch.nn.Sequential(nn.LayerNorm(self.emb_size),
                                nn.Linear(self.emb_size,self.emb_size//4),
                                nn.ReLU(),
                                nn.Linear(self.emb_size//4,self.emb_size),
                                # nn.LayerNorm(self.emb_size)
                )
        self.adapter=torch.nn.Sequential(nn.Linear(self.emb_size,self.emb_size//4),
                                nn.ReLU(),
                                nn.LayerNorm(self.emb_size//4),
                                nn.Linear(self.emb_size//4,self.emb_size),
                                # nn.LayerNorm(self.emb_size)
                )
        self.adapter1.apply(self.init_add_weights)
        self.adapter2.apply(self.init_add_weights)
        self.adapter.apply(self.init_add_weights)
        self.prediction.apply(self.init_add_weights)
        self.prefix_embedding.apply(self.init_embedding)
        self.prefix_q_embedding.apply(self.init_embedding)
        
    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        g_embeddings=nn.Embedding(3,self.emb_size,padding_idx=0)
        y_embeddings=nn.Embedding(self.year_num+1,self.emb_size,padding_idx=0)
        power_embeddings=nn.Embedding(self.power_num+1,self.emb_size,padding_idx=0)
        self.prefix_gen=nn.Linear(self.emb_size*3,self.emb_size)
        side_dict={'g_embeddings':g_embeddings,'y_embeddings':y_embeddings,'power_embeddings':power_embeddings}
        self.side_embeddings=nn.ModuleDict(side_dict)
        if self.encoder_name=='SASRec':
            self.encoder=SASRecEncoder(self.max_his,self.emb_size,self.num_heads,self.num_layers)
        self.feature_gen=nn.Sequential(nn.Linear(self.emb_size*3,self.emb_size),
                nn.ReLU(),
                nn.Linear(self.emb_size,self.emb_size)
            )
        self.cl_lossfun=nn.CrossEntropyLoss()
    def get_user_personlized_prompt(self,feed_dict):
        gender=feed_dict['gender'].clone()
        gender[gender>=0]+=1
        gender[gender<0]=0
        gender_vector=self.gender_embedding(gender)
        age=feed_dict['old'].clone()
        age[age>=0]+=1
        age[age<0]=0
        age_vector=self.age_embedding(age)
        occupation=feed_dict['power'].clone()
        occupation[occupation>=0]+=1
        occupation[occupation<0]=0
        occupation_vector=self.occupation_embedding(occupation)
        return torch.stack([gender_vector,age_vector,occupation_vector],dim=1)
    def get_user_profiles(self,feed_dict,embedding_dcit,cat=True):
        gender_vector=embedding_dcit['gender_embeddings'](feed_dict['gender'])
        age_vector=embedding_dcit['age_embeddings'](feed_dict['age'])
        segid_vecotr=embedding_dcit['segid_embeddings'](feed_dict['segid'])
        groupid_vector=embedding_dcit['groupid_embeddings'](feed_dict['groupid'])
        pvalue_level_vector=embedding_dcit['pvalue_level_embeddings'](feed_dict['pvalue_level'])
        shopping_level_vecotr=embedding_dcit['shopping_level_embeddings'](feed_dict['shopping_level'])
        occupation_vector=embedding_dcit['occupation_embeddings'](feed_dict['occupation'])
        class_level_vector=embedding_dcit['class_level_embeddings'](feed_dict['class_level'])
        if cat: #shifou
            side_info=torch.cat([gender_vector,age_vector,segid_vecotr,groupid_vector,pvalue_level_vector,shopping_level_vecotr,occupation_vector,class_level_vector],dim=1)
        else:
            side_info=[gender_vector,age_vector,segid_vecotr,groupid_vector,pvalue_level_vector,shopping_level_vecotr,occupation_vector,class_level_vector]
        return side_info
    def init_embedding(self,m):
        mean,std=self.i_embeddings.weight.data.mean(),self.i_embeddings.weight.data.std()
        if 'Linear' in str(type(m)):
            nn.init.normal_(m.weight, mean=mean, std=std)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=mean, std=std)
        elif 'Embedding' in str(type(m)):
            nn.init.normal_(m.weight, mean=mean, std=std)
       
    def forward(self, feed_dict):
        side_emb=None
        prefix=None
        adapter=None
        u_ids=feed_dict['user_id']
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = history.shape
        i_vectors = self.i_embeddings(i_ids)
        his_vectors = self.i_embeddings(history)
        
        
        if self.stage==2:
            prefix_len=10
            prefix=[list(range(1+i*prefix_len,1+(i+1)*prefix_len)) for i in range(self.num_layers)]
            prefix=[prefix]*batch_size
            prefix=torch.tensor(prefix).transpose(1,2)
            prefix=prefix.to(his_vectors.device)
            prefix_k=self.prefix_embedding(prefix)
            prefix_q=self.prefix_q_embedding(prefix)
            personlized_prompt=self.get_user_personlized_prompt(feed_dict).unsqueeze(2)
            prefix_k=torch.cat((prefix_k,personlized_prompt),dim=1)
            prefix_q=torch.cat((prefix_q,personlized_prompt),dim=1)
            prefix=prefix_k
            adapter=[self.adapter1,self.adapter2]
            
            # adapter=None
        else:
            adapter=None
            # his_vectors=self.adapter(his_vectors)+his_vectors
        if self.encoder_name=='SASRec': 
            his_vector=self.encoder(his_vectors,lengths,prefix,adapter)
            if self.stage==5:
                his_vector=self.fillter(his_vector)
        if self.stage!=1:
            if self.feature_num==2:
                prediciton_att=self.prediction(his_vector).sigmoid()[:,0]
            else:
                prediciton_att=self.prediction(his_vector).softmax(dim=1)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        
        
        return  {'prediction': prediction.view(batch_size, -1),'prediction_att':prediciton_att,'label':feed_dict[self.predict_profile],'u_vectors':his_vector.detach()}

    def classify_loss(self,u_vecotrs,label):
       
         
        if self.feature_num==2:
            prediction=self.prediction(u_vecotrs).sigmoid()[:,0]
        else:
            prediction=self.prediction(u_vecotrs).softmax(dim=1)
        mask=label>=0
        prediction=prediction[mask]
        label=label[mask]
        if self.feature_num==2:
            c_loss=self.classify_loss_fun(prediction,label.float())
        else:
            c_loss=self.classify_loss_fun(torch.log(prediction),label)
        return c_loss

       

    def dis_loss(self,u_vecotrs,label):
        if self.feature_num==2:
            prediction=self.prediction(u_vecotrs).sigmoid()[:,0].clamp(min=0.1,max=0.9)
        else:
            prediction=self.prediction(u_vecotrs).softmax(dim=1)
        mask=label>=0
        prediction=prediction[mask]
        label=label[mask]
        if self.feature_num==2:
            c_loss=self.c_loss_fun(prediction,label.float())
        else:
            c_loss=self.c_loss_fun(torch.log(prediction),label)
        
        return c_loss
    def loss(self, out_dict: dict) -> torch.Tensor:
        prediction=out_dict['prediction_att']
        label=out_dict['label']
        mask=label>=0
        label=label[mask]
        prediction=prediction[mask]
        if self.feature_num==2:
            c_loss=self.c_loss_fun(prediction,label.float())
        else:
            c_loss=self.c_loss_fun(prediction,label)
        rec_loss=super().loss(out_dict)
        
        return rec_loss,c_loss,-c_loss
        
        
        

    class Dataset(SequentialModel.Dataset):


        def __init__(self, model, corpus, phase: str):
            self.side_info2id={'0':0}
            super().__init__(model, corpus, phase)
        def _prepare(self):
            if self.model.stage==1  : ## Pretrian stage
                
                idx_select = np.array(self.data['position'])>0 # history length must be non-zero
                for key in self.data:
                    self.data[key] = np.array(self.data[key])[idx_select]
            else:

                idx_select =  (np.array(self.data['position'])>7)  # history length must be non-zero
                for key in self.data:
                    self.data[key] = np.array(self.data[key])[idx_select]
            if self.buffer:
                for i in tqdm(range(len(self)), leave=False, desc=('Prepare ' + self.phase)):
                    self.buffer_dict[i] = self._get_feed_dict(i)

        def _get_feed_dict(self, index):


            feed_dict=super()._get_feed_dict(index)
            feed_dict['gender']=self.data['gender'][index]-1
            feed_dict['old']=self.data['old'][index]//10-1
            if feed_dict['old']>=5:
                feed_dict['old']=-1
            feed_dict['power']=self.data['power'][index]-1
            side_tuple=(feed_dict['gender'],feed_dict['old'],feed_dict['power'])
            if side_tuple not in self.side_info2id:
                self.side_info2id[side_tuple]=len(self.side_info2id)
            feed_dict['fid']=self.side_info2id[side_tuple]
            return feed_dict







class SASRecEncoder(nn.Module):
    def __init__(self,max_his,emb_size,num_heads,num_layers,dropout=0.0):
        super().__init__()
        self.p_embeddings = nn.Embedding(max_his + 1, emb_size,padding_idx=0)
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=num_heads,dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self,his_vectors,lengths,prefix=None,adapter=None):
        
        batch_size, seq_len = his_vectors.shape[0],his_vectors.shape[1]
        len_range=torch.arange(seq_len).to(his_vectors.device)
        valid_his = len_range[None, :] < lengths[:, None]
        
        position = (lengths[:, None] - len_range[None, :seq_len]) * valid_his.long()
        
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors
       
        if prefix is not None:
            lengths=lengths+prefix.shape[1]
            seq_len=seq_len+prefix.shape[1]
            len_range=torch.arange(seq_len).to(his_vectors.device)
            valid_his = len_range[None, :] < lengths[:, None]
            his_vectors=torch.cat((prefix[:,:,0,:],his_vectors),dim=1)
        # Self-attention
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(his_vectors.device)
        
        for layer,block in enumerate( self.transformer_block):
            if prefix is not None:
                p=prefix[:,:,layer,:]
                his_vectors=his_vectors[:,prefix.shape[1]:,:]
                his_vectors=torch.cat((p,his_vectors),dim=1)
           
            if adapter is not None:
                his_vectors=block.forward_adapter(his_vectors,attn_mask,adapter[0],adapter[1])

            elif prefix is not None and adapter is not None:
                his_vector=block.forward_adapter_prefix(his_vectors,attn_mask,adapter[0],adapter[1],prefix)
            elif prefix is not None and adapter is None:
                block.forward_prefix(his_vectors,attn_mask,prefix)

            else:
                his_vectors = block(his_vectors, attn_mask)
           
        
        his_vectors = his_vectors * valid_his[:, :, None].float()
        
        his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :]
        # his_vector = his_vectors.sum(1) / lengths[:, None].float()
        # ↑ average pooling is shown to be more effective than the most recent embedding
        return his_vector