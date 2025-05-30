from datasets import load_dataset


class MultiClinSumDatasetLoader(object):

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_from_json(self):
        data_files = {
            'train': f'{self.dataset_path}/train.json',
            'valid': f'{self.dataset_path}/valid.json',
        }
        datasets = load_dataset('json', data_files=data_files)
        return datasets
