# -*- coding: UTF-8 -*-

import os
import gc
from numpy.testing._private.utils import print_assert_equal
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List, NoReturn

from utils import utils
from models.BaseModel import BaseModel
from sklearn.metrics import roc_auc_score
import recbole
from recbole.evaluator.metrics import AUC,GAUC
from recbole.config.configurator import Config
from recbole.model.abstract_recommender import GeneralRecommender


class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=300,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check some tensors every check_epoch.')
        parser.add_argument('--test_epoch', type=int, default=-1,
                            help='Print test results every test_epoch (-1 means no print).')
        parser.add_argument('--early_stop', type=int, default=10,
                            help='The number of epochs when dev results drop continuously.')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='Learning rate.')
        parser.add_argument('--l2', type=float, default=0,
                            help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=256,
                            help='Batch size during testing.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: SGD, Adam, Adagrad, Adadelta')
        parser.add_argument('--num_workers', type=int, default=5,
                            help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=0,
                            help='pin_memory in DataLoader')
        parser.add_argument('--topk', type=str, default='10,5,20,50',
                            help='The number of items recommended to each user.')
        parser.add_argument('--metric', type=str, default='NDCG,HR,AUC',
                            help='metrics: NDCG, HR')
        parser.add_argument('--l_weight', type=float, default=1.0,
                            help='Number of epochs.')
        parser.add_argument('--d_step', type=int, default=5,
                            help='Number of epochs.')
        parser.add_argument('--warm_epoch', type=int, default=0,
                            help='Number of epochs.')
        parser.add_argument('--epoch_step', type=int, default=1000,
                            help='Number of epochs.')
        return parser


    @staticmethod
    def metric_info(pos_rank_sum, user_len_list, pos_len_list):
        """Get the value of GAUC metric.

        Args:
            pos_rank_sum (numpy.ndarray): sum of descending rankings for positive items of each users.
            user_len_list (numpy.ndarray): the number of predicted items for users.
            pos_len_list (numpy.ndarray): the number of positive items for users.

        Returns:
            float: The value of the GAUC.
        """
        neg_len_list = user_len_list - pos_len_list
        # check positive and negative samples
        any_without_pos = np.any(pos_len_list == 0)
        any_without_neg = np.any(neg_len_list == 0)
        non_zero_idx = np.full(len(user_len_list), True, dtype=np.bool)
        if any_without_pos:
            
            print(
                "No positive samples in some users, "
                "true positive value should be meaningless, "
                "these users have been removed from GAUC calculation"
            )
            non_zero_idx *= (pos_len_list != 0)
        if any_without_neg:
           
            print(
                "No negative samples in some users, "
                "false positive value should be meaningless, "
                "these users have been removed from GAUC calculation"
            )
            non_zero_idx *= (neg_len_list != 0)
        if any_without_pos or any_without_neg:
            item_list = user_len_list, neg_len_list, pos_len_list, pos_rank_sum
            user_len_list, neg_len_list, pos_len_list, pos_rank_sum = map(lambda x: x[non_zero_idx], item_list)

        pair_num = (user_len_list + 1) * pos_len_list - pos_len_list * (pos_len_list + 1) / 2 - np.squeeze(pos_rank_sum)
        user_auc = pair_num / (neg_len_list * pos_len_list)
        result = (user_auc * pos_len_list).sum() / pos_len_list.sum()
        return result


    @staticmethod
    def evaluate_method(predictions: Dict[str,np.ndarray], topk: list, metrics: list) -> Dict[str, float]:
        """
        :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
        :param topk: top-K value list
        :param metrics: metric string list
        :return: a result dict, the keys are metric@topk
        """
        evaluations = dict()
        # print(predictions[0])
        batch_size,score_len=predictions.shape[0],predictions.shape[1]
        if score_len<1000:
            predictions=predictions.numpy()
            sort_idx = (-predictions).argsort(axis=1)
            gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
        else:
            
            user_lens=(predictions!=-np.inf).sum(axis=1)
            # sort_idx = (-predictions).argsort(dim=1)
            # gt_rank=torch.where(sort_idx==0)[1]+1
            gt_rank=(predictions>predictions[:,0][:,None]).sum(dim=1)+1
            gt_rank=gt_rank.numpy()
            
            user_lens=user_lens.numpy()
        # if 'AUC' in metrics:
        #     batch_size,score_len=sort_idx.shape[0],sort_idx.shape[1]
        #     user_lens=np.array([score_len]*batch_size)
        #     pos_len_list=np.array([1]*batch_size)
        #     # gauc=GAUC(Config(model='FM', dataset='ml-100k')) #just a model that can init GAUC no use
          
        #     auc_score=BaseRunner.metric_info(gt_rank,user_lens,pos_len_list)
            
            # evaluations['AUC']=auc_score
        if 'AUC' in metrics:
            if -np.inf in predictions[0]:
                pos_len_list=np.array([1]*batch_size)
                auc_score=BaseRunner.metric_info(gt_rank,user_lens,pos_len_list)
                evaluations['AUC']=auc_score
            else:
                auc_score=((100-gt_rank)/99).mean()
                evaluations['AUC']=auc_score

       
        
        for k in topk:
            hit = (gt_rank <= k)
            for metric in metrics:
                if metric == 'HR':
                    key = '{}@{}'.format(metric, k)
                    evaluations[key] = hit.mean()
                elif metric == 'NDCG':
                    key = '{}@{}'.format(metric, k)
                    evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
                else:
                    pass
                    # raise ValueError('Undefined evaluation metric: {}.'.format(metric))
        
        return evaluations

    def __init__(self, args):
        self.epoch = args.epoch
        self.check_epoch = args.check_epoch
        self.test_epoch = args.test_epoch
        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.topk = [int(x) for x in args.topk.split(',')]
        self.metrics = [m.strip().upper() for m in args.metric.split(',')]
        if self.metrics[0]!='AUC':
            self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0])  # early stop based on main_metric
        else:
            self.main_metric=self.metrics[0]
        self.time = None  # will store [start_time, last_step_time]

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        logging.info('Optimizer: ' + self.optimizer_name)
        optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(
            model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        return optimizer

    def train(self, data_dict: Dict[str, BaseModel.Dataset]) -> NoReturn:
        model = data_dict['train'].model
        main_metric_results, dev_results = list(), list()
        self._check_time(start=True)
        try:
            for epoch in range(self.epoch):
                # Fit
                self._check_time()
                gc.collect()
                torch.cuda.empty_cache()
                loss = self.fit(data_dict['train'], epoch=epoch + 1)
                training_time = self._check_time()

                # Observe selected tensors
                if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
                    utils.check(model.check_list)

                # Record dev results
                dev_result = self.evaluate(data_dict['dev'], self.topk[:1], self.metrics)
                dev_results.append(dev_result)
                main_metric_results.append(dev_result[self.main_metric])
                logging_str = 'Epoch {:<5} loss={:<.4f} [{:<3.1f} s]    dev=({})'.format(
                    epoch + 1, loss, training_time, utils.format_metric(dev_result))
               
                # Test
                if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
                    
                    test_result = self.evaluate(data_dict['test'], self.topk[:1], self.metrics)
                    logging_str += ' test=({})'.format(utils.format_metric(test_result))
                testing_time = self._check_time()
                logging_str += ' [{:<.1f} s]'.format(testing_time)

                # Save model and early stop
                if max(main_metric_results) == main_metric_results[-1] or \
                        (hasattr(model, 'stage') and model.stage == 4):
                    model.save_model()
                    logging_str += ' *'
                
                logging.info(logging_str)
               
                if self.early_stop > 0 and self.eval_termination(main_metric_results):
                    logging.info("Early stop at %d based on dev result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)

        # Find the best dev result across iterations
        best_epoch = main_metric_results.index(max(main_metric_results))
        logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
            best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
        model.load_model()

    def fit(self, data: BaseModel.Dataset, epoch=-1) -> float:
        model = data.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        data.actions_before_epoch()  # must sample before multi thread start

        model.train()
        loss_lst = list()
        dl = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, model.device)
            model.optimizer.zero_grad()
            out_dict = model(batch)
            loss = model.loss(out_dict)
            loss.backward()
            model.optimizer.step()
            loss_lst.append(loss.detach().cpu().data.numpy())
            
            
            # print(loss.item())
            
        return np.mean(loss_lst).item()

    def eval_termination(self, criterion: List[float]) -> bool:
       
        if len(criterion) > self.early_stop+10 and utils.non_increasing(criterion[-self.early_stop:]):
           
            return True
        elif len(criterion) - criterion.index(max(criterion)) > self.early_stop+10:
            return True
        return False

    def get_metric(pred_list, topk=10):
        NDCG = 0.0
        HIT = 0.0
        MRR = 0.0
        AUC=0.0
        # [batch] the answer's rank
        for rank in pred_list:
            MRR += 1.0 / (rank + 1.0)
            AUC+=(100-rank-1)/99
            if rank < topk:
                NDCG += 1.0 / np.log2(rank + 2.0)
                HIT += 1.0
           
        return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list),AUC/len(pred_list)


    

    def evaluate(self, data: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
        """
        Evaluate the results for an input dataset.
        :return: result dict (key: metric@k)
        """
        predictions = self.predict(data)
        print(predictions.mean(),predictions.std())
        if data.model.test_all:
            rows, cols = list(), list()
            for i, u in enumerate(data.data['user_id']):
                clicked_items = [x[0] for x in data.corpus.user_his[u]]
                # clicked_items = [data.data['item_id'][i]]
                idx = list(np.ones_like(clicked_items) * i)
                rows.extend(idx)
                cols.extend(clicked_items)
            predictions[rows, cols] = -np.inf
        print((predictions==-np.inf).sum())
        return self.evaluate_method(predictions, topks, metrics)

    def predict(self, data: BaseModel.Dataset) -> np.ndarray:
        """
        The returned prediction is a 2D-array, each row corresponds to all the candidates,
        and the ground-truth item poses the first.
        Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
                 predictions like: [[1,3,4], [2,5,6]]
        """
        data.model.eval()
       
        predictions = list()
        dl = DataLoader(data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        
        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
            prediction = data.model.inference(utils.batch_to_gpu(batch, data.model.device))
            prediction=prediction['prediction']
            # print(prediction)
            predictions.append(prediction.cpu().data)
           
        
        predictions=torch.cat(predictions,dim=0)
        return predictions

    def print_res(self, data: BaseModel.Dataset) -> str:
        """
        Construct the final result string before/after training
        :return: test result string
        """
        result_dict = self.evaluate(data, self.topk, self.metrics)
        res_str = '(' + utils.format_metric(result_dict) + ')'
        return res_str
