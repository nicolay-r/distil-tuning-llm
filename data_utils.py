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


# class DatasetLoader(object):
#     def __init__(self, dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
#                  batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None):
#         self.data_root = DATASET_ROOT
#         self.dataset_name = dataset_name
#         self.dataset_version = dataset_version
#         self.has_valid = has_valid
#         self.split_map = split_map

#         self.batch_size = batch_size
#         self.train_batch_idxs = train_batch_idxs
#         self.test_batch_idxs = test_batch_idxs
#         self.valid_batch_idxs = valid_batch_idxs
        
#         assert self.split_map is not None    


#     # def load_from_source(self):
#     #     if self.source_dataset_name is None:
#     #         self.source_dataset_name = self.dataset_name
#     #     if self.dataset_version is None:
#     #         datasets = load_dataset(self.source_dataset_name)
#     #     else:
#     #         datasets = load_dataset(self.source_dataset_name, self.dataset_version)
#     #     return datasets


#     def to_json(self, datasets):
#         for k, v in self.split_map.items():
#             datasets[v].to_json(f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_{k}.json')


#     def load_from_json(self):
#         data_files = {
#             'train': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_train.json',
#             'test': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_test.json',
#         }

#         if self.has_valid:
#             data_files.update({'valid': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_valid.json',})
        
#         datasets = load_dataset('json', data_files=data_files) # 这行报错，所以改为下面方法
        
#         datasets = self._post_process(datasets) 

#         # subsample training dataset if needed
#         num_train = len(datasets['train'])
#         idxs = list()
#         for idx in self.train_batch_idxs:
#             idxs += range(idx*self.batch_size, (idx+1)*self.batch_size)        
#         datasets['train'] = Dataset.from_dict(datasets['train'][[idx for idx in idxs if idx < num_train]])

#         return datasets


#     def load_llm_preds(self, split):
#         labels = list()
#         rationales = list()
#         for idx in getattr(self, f'{split}_batch_idxs'):
#             with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json') as f:
#                 outputs = json.load(f)

#             for output in outputs:
#                 rationale, label = self._parse_llm_output(output)

#                 rationales.append(rationale)
#                 labels.append(label)

#         return rationales, labels


#     def load_gpt_preds(self, split):
#         labels = list()
#         rationales = list()
        
#         with open(f'{self.data_root}/gpt-neox/{self.dataset_name}/{split}.json') as f:
#             outputs = json.load(f)

#         for output in outputs:
#             rationale, label = self._parse_gpt_output(output)

#             rationales.append(rationale)
#             labels.append(label)

#         return rationales, labels


#     def _post_process(self, datasets):
#         raise NotImplementedError


#     def _parse_llm_output(self, output):
#         raise NotImplementedError


#     def _parse_gpt_output(self, output):
#         raise NotImplementedError

#xiaoxiao liu
class MEDQADatasetLoader(object):
    def __init__(self, dataname, model_type):
        self.dataset_name = dataname
        self.data_root = DATASET_ROOT
        self.dataset_version = None
        self.has_valid = True
        self.split_map = {
            'train': 'train',
            'test': 'test',
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
            'test': f'{self.data_root}/{self.dataset_name}/{self.model_type}/{self.dataset_name}_test.json',
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
            'test': f'{self.data_root}/{self.dataset_name}/{self.model_type}/{self.dataset_name}_test.json',
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



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, required=True)
#     args = parser.parse_args()

#     if args.dataset == 'cqa':
#         dataset_loader = CQADatasetLoader()
#     elif args.dataset == 'svamp':
#         dataset_loader = SVAMPDatasetLoader()
#     elif args.dataset == 'esnli':
#         dataset_loader = ESNLIDatasetLoader()
#     elif args.dataset == 'anli1':
#         dataset_loader = ANLI1DatasetLoader()

#     datasets = dataset_loader.load_from_source()
#     dataset_loader.to_json(datasets)
