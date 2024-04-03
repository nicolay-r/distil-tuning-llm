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
        self.batch_size = 500
        self.train_batch_idxs = range(2)
        self.test_batch_idxs = range(1)
        self.model_type = model_type


        # super().__init__(dataset_name, dataset_version, has_valid, split_map,
                #  batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)

    def load_from_json(self):
        data_files = {
            'train': f'{self.data_root}/{self.dataset_name}/{self.model_type}/{self.dataset_name}_train.json',
            # 'test': f'{self.data_root}/{self.dataset_name}/{self.model_type}/{self.dataset_name}_test.json',
        }

        if self.has_valid:
            data_files.update({'valid': f'{self.data_root}/{self.dataset_name}/{self.model_type}/{self.dataset_name}_valid.json',})
        
        # breakpoint()
        datasets = load_dataset('json', data_files=data_files)
        
        datasets = self._post_process(datasets) 

        return datasets

    
    def load_from_json_rationale(self):
        data_files = {
            'train': f'{self.data_root}/{self.dataset_name}/{self.model_type}/{self.dataset_name}_train.json',
            # 'test': f'{self.data_root}/{self.dataset_name}/{self.model_type}/{self.dataset_name}_test.json',
        }

        if self.has_valid:
            data_files.update({'valid': f'{self.data_root}/{self.dataset_name}/{self.model_type}/{self.dataset_name}_valid.json',})
        
        # breakpoint()
        datasets = load_dataset('json', data_files=data_files) # 这行报错，所以改为下面方法
        
        datasets = self._post_process(datasets) 

        return datasets

    
    def load_rationale_data(self, split):
        labels = list()
        rationales = list()
        with open(f'{self.data_root}/{self.dataset_name}/{self.model_type}/{self.dataset_name}_{split}.json') as f:
            outputs = json.load(f)
            
        for output in outputs:
            rationale = output['rationale']
            label = output['output']
            rationales.append(rationale)
            labels.append(label)
        # breakpoint()
        return rationales, labels
    
    def load_gt_preds(self, split):
        labels = list()
        rationales = list()
        
        with open(f'{self.data_root}/{self.dataset_name}/{self.model_type}/{self.dataset_name}_{split}.json') as f:
            outputs = json.load(f)
        
        for output in outputs:
            rationale = ""
            label = output['output']

            rationales.append(rationale)
            labels.append(label)
        # breakpoint()
        return labels
        

    def _post_process(self, datasets):
        return datasets


    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        try:
            rationale, label = rationale_label.split('The answer is')
        except:
            rationale = ' '
            label = ' '
            return rationale, label
            
        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label

    def _parse_gpt_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip().lstrip()
        try:
            rationale, label = rationale_label.split('The answer is')
        except:
            rationale = ' '
            label = ' '
            return rationale, label
            
        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label


