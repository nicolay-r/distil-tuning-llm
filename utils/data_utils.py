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


import json
from os.path import dirname, realpath, join
from datasets import load_dataset


class MultiClinSumDatasetLoader(object):

    def __init__(self, dataname):
        current_dir = dirname(realpath(__file__))
        self.dataset_name = dataname
        self.data_root = join(current_dir, "../datasets/")

    def path_to_split(self, split):
        return f'{self.data_root}/{self.dataset_name}/{split}.json'

    def load_from_json(self):
        data_files = {
            'train': self.path_to_split(split="train"),
            'valid': self.path_to_split(split="valid"),
        }
        datasets = load_dataset('json', data_files=data_files)
        return datasets

    def load_rationale_data(self, split):
        labels = list()
        rationales = list()
        with open(self.path_to_split(split)) as f:
            outputs = json.load(f)
            
        for output in outputs:
            rationale = output['rationale']
            label = output['output']
            rationales.append(rationale)
            labels.append(label)
        # breakpoint()
        return rationales, labels

    