# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import re
import json
import numpy as np

import pandas as pd
# from datasets import Dataset
from datasets import Dataset, DatasetDict, load_dataset


DATASET_ROOT = 'datasets'

class MEDQADatasetLoader(object):
    def __init__(self, dataname, model_type):
        self.dataset_name = dataname
        self.data_root = DATASET_ROOT
        self.dataset_version = None
        self.has_valid = True
        self.split_map = {
            'train': 'train',
            'valid': 'valid'
            # 'test': 'test',
        }
        # self.batch_size = 500
        # self.train_batch_idxs = range(2)
        # self.test_batch_idxs = range(1)
        self.model_type = model_type

    def load_from_json(self):
        data_files = {
            'train': f'../{self.data_root}/{self.dataset_name}/{self.model_type}/{self.dataset_name}_train.json',
            'valid': f'../{self.data_root}/{self.dataset_name}/{self.model_type}/{self.dataset_name}_valid.json',
        }
        datasets = load_dataset('json', data_files=data_files)       

        return datasets

    
    def load_from_json_rationale(self):
        data_files = {
            # 'train': f'../{self.data_root}/{self.dataset_name}/{self.model_type}/{self.dataset_name}_train.json',
            'valid': f'../{self.data_root}/{self.dataset_name}/{self.model_type}/{self.dataset_name}_valid.json',
        }
        breakpoint()
        datasets = load_dataset('json', data_files=data_files) 
        # breakpoint()
        return datasets

    
    def load_rationale_data(self, split):
        labels = list()
        rationales = list()
        with open(f'../{self.data_root}/{self.dataset_name}/{self.model_type}/{self.dataset_name}_{split}.json') as f:
            outputs = json.load(f)
            
        for output in outputs:
            rationale = output['rationale']
            label = output['output']
            rationales.append(rationale)
            labels.append(label)
        # breakpoint()
        return rationales, labels
    
    def load_multi_rationale(self, split):
        labels = list()
        rationales = list()
        with open(f'../{self.data_root}/{self.dataset_name}/{self.model_type}/{self.dataset_name}_{split}.json') as f:
            outputs = json.load(f)
        breakpoint()    
        i = 0
        for output in outputs:
            print(i)
            rationale = output['rationale']
            label = output['output']
            # rationales.append(rationale)
            labels.append(label)
            i+=1
        breakpoint()
        return rationales, labels
    