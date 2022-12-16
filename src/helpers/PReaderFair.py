from helpers.BaseReader import BaseReader
import logging
from typing import NoReturn
from utils import utils
import os
import pandas as pd
import numpy as np

class PReaderFair(BaseReader):

    
      
    
    def __init__(self, args):
        super().__init__(args)
        
    def _read_data(self) -> NoReturn:
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        for key in ['train', 'dev', 'test','trainc','devc','testc']:
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep)
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])
        
        logging.info('Counting pretraining dataset statistics...')
        self.all_df = pd.concat([df[['user_id', 'item_id', 'time','old','gender','power']] for df in self.data_df.values()])
        self.n_users, self.n_items = len(self.all_df['user_id'].unique()), len(self.all_df['item_id'].unique())
        logging.info('"pretraining # user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users, self.n_items, len(self.all_df)))

        
        
      
        logging.info('"pretraining # user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users , self.n_items , len(self.all_df)))

        logging.info('Counting total dataset statistics...')
        self.all_df = pd.concat([df[['user_id', 'item_id', 'time','old','gender','power']] for df in self.data_df.values()])
        self.n_users, self.n_items,self.n_years,self.n_genders,self.n_powers = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1,self.all_df['old'].max()+1 ,self.all_df['gender'].max() + 1,self.all_df['power'].max() + 1
        
                
        logging.info('"total # user": {}, "# item": {}, "# entry": {}, "# gender": {}, "# old": {}, "# power": {}'.format(
            self.n_users - 1, self.n_items - 1, len(self.all_df), self.n_genders - 1,self.n_years - 1,self.n_powers-1))




    def _append_his_info(self) -> NoReturn:
        """
        Add history info to data_df: position
        ! Need data_df to be sorted by time in ascending order
        """

        
        logging.info('Appending history info...')
        self.user_his = dict()  # store the already seen sequence of each user
        self.train_clicked_set = dict()  # store the clicked item set of each user in training set
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            position = list()
            for uid, iid, t in zip(df['user_id'], df['item_id'], df['time']):
                if uid not in self.user_his:
                    self.user_his[uid] = list()
                    self.train_clicked_set[uid] = set()
                position.append(len(self.user_his[uid]))
                self.user_his[uid].append((iid, t))
                if 'train' in key:
                    self.train_clicked_set[uid].add(iid)
            df['position'] = position


        
        for key in ['trainc','devc','testc']:
            df = self.data_df[key]
            position = list()
            for uid, iid, t in zip(df['user_id'], df['item_id'], df['time']):
                    position.append(len(self.user_his[uid])-1)

            df['position'] = position

       

        