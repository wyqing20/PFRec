
from helpers.BaseRunner import BaseRunner
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
from recbole.evaluator.metrics import AUC,GAUC, Precision
from recbole.config.configurator import Config
from recbole.model.abstract_recommender import GeneralRecommender


class FairRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        self.main_metric='AUC'
        self.l_weight=args.l_weight
        self.d_step=args.d_step
        self.warm_epoch=args.warm_epoch
        self.epoch_step=args.epoch_step
    def _build_optimizer(self, model,type):
        logging.info('Optimizer: ' + self.optimizer_name)
        if type=='d': # 判别器的optimizer
            optimizer = eval('torch.optim.{}'.format('RMSprop'))(
                model.d_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        elif type=='g': # 生成器的optimizer
            optimizer = eval('torch.optim.{}'.format('RMSprop'))(
                model.g_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        elif type=='all':
            optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(
                    model.all_parameters(), lr=self.learning_rate, weight_decay=self.l2) 
        elif type=='c':
            optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(
                model.d_parameters(), lr=self.learning_rate, weight_decay=self.l2) 
        return optimizer
    def train(self, data_dict: Dict[str, BaseModel.Dataset],classify=0) -> NoReturn:
        model = data_dict['train'].model
        main_metric_results, dev_results = list(), list()
        self._check_time(start=True)
        res=1.0
        try:
            for epoch in range(self.epoch):
                # Fit
                self._check_time()
                gc.collect()
                torch.cuda.empty_cache()
                # 这块逻辑我有点忘了
                if model.stage!=1:
                    if  classify==0 and model.stage!=4:
                        loss,c_loss,n_c_loss,u_loss = self.fit(data_dict['train'], epoch=epoch + 1,classify=False)
                    elif classify==1 or model.stage==4:
                        loss,c_loss,n_c_loss,u_loss = self.fit_classify(data_dict['trainc'], epoch=epoch + 1,classify=True)
                    elif classify==2:
                        loss,c_loss,n_c_loss,u_loss = self.fit_classify(data_dict['train'], epoch=epoch + 1,classify=True)

                else:
                    
                    loss,c_loss,n_c_loss = self.fit_pretrain(data_dict['train'], epoch=epoch + 1,classify=classify)
                training_time = self._check_time()

                # Observe selected tensors
                if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
                    utils.check(model.check_list)

                # Record dev results
                if  (classify==0 or classify==2) and model.stage!=4:
                    dev_result,dev_e2 = self.evaluate(data_dict['dev'], self.topk[:1], self.metrics)
                else:
                    dev_result,dev_e2 = self.evaluate(data_dict['devc'], self.topk[:1], self.metrics)
                if model.stage!=1:
                    dev_results.append(dev_result)
                    if classify or model.stage==4:
                        main_metric_results.append(dev_result[self.main_metric])
                    else:
                        res-=0.1
                        main_metric_results.append(-res)
                else:
                    dev_results.append(dev_result)
                    main_metric_results.append(dev_result['NDCG@10'])
                logging_str = 'Epoch {:<5} loss={:<.4f} c_loss={:<.4f} n_c_loss={:<.4f} u_loss={:<.4f} [{:<3.1f} s]    dev=({}  {})'.format(
                    epoch + 1, loss,c_loss,n_c_loss,u_loss, training_time, utils.format_metric(dev_result),utils.format_metric(dev_e2))
               
                # Test
                if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
                    
                    test_result = self.evaluate(data_dict['test'], self.topk[:1], self.metrics)
                    logging_str += ' test=({} {})'.format(utils.format_metric(test_result),utils.format_metric(dev_e2))
                testing_time = self._check_time()
                logging_str += ' [{:<.1f} s]'.format(testing_time)
               
                # Save model and early stop
                if max(main_metric_results) == main_metric_results[-1] or \
                        (hasattr(model, 'stage') and model.stage == -1):
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

    def fit_pretrain(self, data: BaseModel.Dataset, epoch=-1,classify=False):
        model = data.model
        loss_lst=[]
        if model.d_optimizer is None:
            model.d_optimizer = self._build_optimizer(model,type='d')
            if model.stage==2 or model.stage==4:
                model.g_optimizer=  self._build_optimizer(model,type='g')
            else:
                model.g_optimizer=self._build_optimizer(model,type='all')
        data.actions_before_epoch()
        dl = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
           
            batch = utils.batch_to_gpu(batch, model.device)
            out_dict = model(batch)
            rec_loss,c_loss,n_c_loss = model.loss(out_dict)
            model.g_optimizer.zero_grad()
            rec_loss.backward()
            model.g_optimizer.step()
            loss_lst.append(rec_loss.detach().cpu().data.numpy())
        return np.mean(loss_lst).item(),0.0,0.0
    
    def fit(self, data: BaseModel.Dataset, epoch=-1,classify=False) -> float:
        '''
        fit 函数 classify 代表是否是分类
        '''
        model = data.model
        if model.d_optimizer is None:
            model.d_optimizer = self._build_optimizer(model,type='d')
            if model.stage==2 or model.stage==4 or model.stage==5 or model.stage==6:
                model.g_optimizer=  self._build_optimizer(model,type='g')
            else:
                model.g_optimizer=self._build_optimizer(model,type='all')
        data.actions_before_epoch()  # must sample before multi thread start
        
        model.train()
        loss_lst = list()
        c_loss_lst = list()
        n_c_loss_lst = list()
        u_loss_lst=list()
        dl = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        
       
        i=0
        if model.stage==4:
            cc=1
        else:
            cc=1
        for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            i+=1
            batch = utils.batch_to_gpu(batch, model.device)
            out_dict = model(batch)
            
            if  (model.stage==2 or model.stage==3 or model.stage==5 or model.stage==6) and not classify and i%self.d_step==0 and epoch>0:# 这里每d_step我们梯度一步生成器
                
                rec_loss,c_loss,n_c_loss = model.loss(out_dict)
                n_c_loss_lst.append(n_c_loss.detach().cpu().data.numpy())
                loss_lst.append(rec_loss.detach().cpu().data.numpy())
                n_c_loss=n_c_loss*self.l_weight+rec_loss
                u_loss_lst.append(n_c_loss.detach().cpu().data.numpy())
                model.g_optimizer.zero_grad()
                n_c_loss.backward()
                model.g_optimizer.step()
            else:
                if  classify:
                    cc=1
                if i%1==0: #这里我们每一步都调节判别器，每次走cc步这里是一步
                    for _ in range(cc):
                    
                        model.d_optimizer.zero_grad()
                        c_loss = model.dis_loss(out_dict['u_vectors'],out_dict['label'])
                        c_loss_lst.append(c_loss.detach().cpu().data.numpy())
                        c_loss.backward()
                        model.d_optimizer.step()
            

            
            if i>self.epoch_step: #这里固定 epoch_step来做
                break
        if not classify and model.stage!=4:  
            return np.mean(loss_lst).item(),np.mean(c_loss_lst).item(),np.mean(n_c_loss_lst),np.mean(u_loss_lst)
        else:
            return 0.0,np.mean(c_loss_lst).item(),0.0,0.0

    def fit_classify(self, data: BaseModel.Dataset, epoch=-1,classify=False) -> float: 
        '''
        
        '''
        model = data.model
        print('fit_classify')
        if model.d_optimizer is None:
            model.d_optimizer = self._build_optimizer(model,type='c')
            if model.stage==2 or model.stage==4 or model.stage==5:
                model.g_optimizer=  self._build_optimizer(model,type='g')
            else:
                model.g_optimizer=self._build_optimizer(model,type='all')
        data.actions_before_epoch()  # must sample before multi thread start
        
        model.train()
        loss_lst = list()
        c_loss_lst = list()
        n_c_loss_lst = list()
        u_loss_lst=list()
        dl = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        i=0
   
       
        i=0
        if model.stage==4:
            cc=1
        else:
            cc=1
        for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            i+=1
            batch = utils.batch_to_gpu(batch, model.device)
            out_dict = model(batch)
            
            
                
            if i%1==0:
                for _ in range(cc):
                   
                    model.d_optimizer.zero_grad()
                    # out_dict = model(batch)
                    
                    c_loss = model.classify_loss(out_dict['u_vectors'],out_dict['label'])
                    c_loss_lst.append(c_loss.detach().cpu().data.numpy())
                    c_loss.backward()
                    model.d_optimizer.step()
            

            
            if i>1000:
                break
        if not classify and model.stage!=4:  
            return np.mean(loss_lst).item(),np.mean(c_loss_lst).item(),np.mean(n_c_loss_lst),np.mean(u_loss_lst)
        else:
            return 0.0,np.mean(c_loss_lst).item(),0.0,0.0

    def evaluate(self, data: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
        """
        Evaluate the results for an input dataset.
        :return: result dict (key: metric@k)
        """
        predictions,prediction_atts,labels = self.predict(data)
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
        if data.model.stage!=1:
            e1= self.classify_eval(prediction_atts,labels)
            e2= self.evaluate_method(predictions, topks, metrics)
            return e1,e2
        else:
            return self.evaluate_method(predictions, topks, metrics),{}
    
    def predict(self, data: BaseModel.Dataset) -> np.ndarray:
        """
        The returned prediction is a 2D-array, each row corresponds to all the candidates,
        and the ground-truth item poses the first.
        Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
                 predictions like: [[1,3,4], [2,5,6]]
        """
        data.model.eval()
        labels=[]
        prediction_atts = list()
        predictions=list()
        dl = DataLoader(data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        
        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
            prediction = data.model.inference(utils.batch_to_gpu(batch, data.model.device))
            label=prediction['label']
            prediction_att=prediction['prediction_att']
            prediction=prediction['prediction']
            if data.model.stage!=1:
                prediction_atts.append(prediction_att.cpu().data)
            labels.append(label.cpu().data)
            predictions.append(prediction.cpu().data)
        predictions=torch.cat(predictions,dim=0)
        if data.model.stage!=1:
            prediction_atts=torch.cat(prediction_atts,dim=0)
        labels=torch.cat(labels,dim=0)
        return predictions,prediction_atts,labels

    def classify_eval(self,prediction,labels):
        from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,roc_auc_score
        from sklearn.metrics import confusion_matrix,classification_report
        mask=labels>=0
        prediction=prediction[mask]
        labels=labels[mask]
        print(prediction)
        if labels.max()==1:
            self.main_metric='AUC'
            auc=roc_auc_score(labels,prediction)
            prediction=(prediction>=0.5).long()
        else:
            self.main_metric='acc'
            auc=roc_auc_score(labels,prediction,multi_class='ovo')
            prediction=torch.argmax(prediction,dim=1)
        if auc<0.5:
            auc=1-auc
        # logging.info(classification_report(labels,prediction))
        print(classification_report(labels,prediction))
        acc=f1_score(labels, prediction,average='micro')
        f1_macro=f1_score(labels, prediction,average='macro')
        return {'acc':acc,'f1_macro':f1_macro,'AUC':auc}
    def print_res(self, data: BaseModel.Dataset) -> str:
        """
        Construct the final result string before/after training
        :return: test result string
        """
        result_dict,e2 = self.evaluate(data, self.topk, self.metrics)
        res_str = '(' + utils.format_metric(result_dict) + ' ,'+ utils.format_metric(e2)+')'
        return res_str



   