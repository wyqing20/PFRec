
import profile
from helpers.FairRunner import FairRunner
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


class FairRunnerCIKM2(FairRunner):

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
            e1= self.classify_eval(prediction_atts,labels,data.model.prediction_mask)
            e2= self.evaluate_method(predictions, topks, metrics)
            return e1,e2
        else:
            return self.evaluate_method(predictions, topks, metrics),


    def classify_eval(self,predictions,labels,prediction_masks=None):
        from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,roc_auc_score
        from sklearn.metrics import confusion_matrix,classification_report
        aucs,accs,f1_macros=list(),list(),list()
        s=''
        for profile,prediction,label, prediction_mask in zip(['gender','old','power'],predictions,labels,prediction_masks):
            if prediction_mask==0:
                continue
            mask=label>=0
            label=label[mask]
            prediction=prediction[mask]
            # prediction_labels=torch.argmax(prediction,dim=1)
            
          
            if label.max()==1: #二分类
                print(prediction)
                print(label)
                # self.main_metric='AUC'
                prediction_label=(prediction>=0.5).long()
                # print(prediction[torch.arange(0,prediction.shape[0]),labels])
                auc=roc_auc_score(label,prediction)
                acc=f1_score(label, prediction_label,average='micro')
                f1_macro=acc

            else: #多分类
                # self.main_metric='acc'
                print(prediction.shape,label.shape)
                prediction_label=torch.argmax(prediction,dim=1)
                auc=roc_auc_score(label,prediction,multi_class='ovo')
                acc=f1_score(label, prediction_label,average='micro')
                f1_macro=f1_score(label, prediction_label,average='macro')

            if auc<0.5:
                auc=1-auc
            aucs.append(auc)
            accs.append(acc)
            f1_macros.append(f1_macro)
            print(classification_report(label,prediction_label))
            s=s+profile +' auc: {:<.4f} acc: {:<.4f} f1_macros: {:<.4f}\n'.format(auc,acc,f1_macro)
       
      
        logging.info(s)
        
        return {'acc':acc,'f1_macro':f1_macro,'AUC':auc}

    def predict(self, data: BaseModel.Dataset) -> np.ndarray:
        """
        The returned prediction is a 2D-array, each row corresponds to all the candidates,
        and the ground-truth item poses the first.
        Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
                 predictions like: [[1,3,4], [2,5,6]]
        """
        data.model.eval()
        labels=[[],[],[]]
        prediction_atts = list()
        predictions=list()
        prediction_atts=[[],[],[]]
        dl = DataLoader(data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        
        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
            prediction = data.model.inference(utils.batch_to_gpu(batch, data.model.device))
            label=prediction['label']
            prediction_att=prediction['prediction_att']
            prediction_att=[one_profile.cpu().data for one_profile in prediction_att]
            prediction=prediction['prediction']
            predictions.append(prediction.cpu().data)
            label=[one_profile.cpu().data for one_profile in label]
            for i in range(len(prediction_atts)):
                labels[i].append(label[i])
                prediction_atts[i].append(prediction_att[i])
        predictions=torch.cat(predictions,dim=0)
        if data.model.stage!=1:
            prediction_atts=[torch.cat(prediction_att,dim=0) for prediction_att in prediction_atts]
        labels=[torch.cat(label,dim=0) for label in labels]
        return predictions,prediction_atts,labels