
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


class FairRunnerCIKM(FairRunner):

    def classify_eval(self,prediction,labels):
        from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,roc_auc_score
        from sklearn.metrics import confusion_matrix,classification_report
        mask=labels>=0
        labels=labels[mask]
        prediction=prediction[mask]
        # prediction_labels=torch.argmax(prediction,dim=1)
        
        print(prediction)
        print(labels)
        if labels.max()==1: #二分类
            self.main_metric='AUC'
            prediction_labels=(prediction>=0.5).long()
            # print(prediction[torch.arange(0,prediction.shape[0]),labels])
            auc=roc_auc_score(labels,prediction)
            acc=f1_score(labels, prediction_labels)
            f1_macro=acc

        else: #多分类
            self.main_metric='acc'
            print(labels.shape,prediction.shape)
            prediction_labels=torch.argmax(prediction,dim=1)
            auc=roc_auc_score(labels,prediction,multi_class='ovo')
            acc=f1_score(labels, prediction_labels,average='micro')
            f1_macro=f1_score(labels, prediction_labels,average='macro')

        if auc<0.5:
            auc=1-auc
        # print(prediction[:20])
        # print(label[:20])
        # print(((prediction[:20]-label[:20])!=0).sum())
        print(classification_report(labels,prediction_labels))
        
        return {'acc':acc,'f1_macro':f1_macro,'AUC':auc}