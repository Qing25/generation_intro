

import os
import sys
from typing import Text
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
# sys.path.append(PROJ_DIR)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from transformers import AutoTokenizer, BertTokenizerFast, BartTokenizer, T5Tokenizer
from transformers import BertTokenizer

import pytorch_lightning as pl
from dataclasses import dataclass
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split
from q_snippets.data import sequence_padding, BaseData, save_json, load_json
from q_snippets.object import Config, print_config, print_string


@dataclass
class TextSample:
    """
        for pre-training
        one line, one sentence 
    """
    text: str 
 


@dataclass
class TextFeature:
    input_ids : list 
    attention_mask : list 
    token_type_ids : list 

    @classmethod
    def from_tokenized(cls, td):
        return cls(td.input_ids, td.attention_mask, td.token_type_ids)


@dataclass
class TextBatch(BaseData):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor

    def __len__(self):
        return self.input_ids.size(0)


class DataReader:

    def _get_max_len(self, features):
        """ 
        """
        max_len = 0
        for feature in features:
            if len(feature.input_ids) > max_len:
                max_len = len(feature.input_ids) 

        if max_len % 1:
            max_len = max_len + 1
        return max_len

    def load_samples(self, path, mode):
        samples = []
        with open(path, 'r', encoding='utf8') as f:
            for line in tqdm(f):
                sent = line.strip()
                if sent:
                    _sample = TextSample(sent)
                    samples.append(_sample)
        return samples


    def load_features(self, samples, tokenizer):
        """ 

        """
        features = []
        for sample in samples:
            input_td = tokenizer(sample.text)   
            feature = TextFeature.from_tokenized(input_td)
            features.append(feature)
        return features




class TextDataset(Dataset):
    def __init__(self, config, tokenizer, dataset_mode) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.mode = dataset_mode
        self.datareader = DataReader()
        self.samples, self.features = self._handle_cache()
        self.max_len = self.datareader._get_max_len(self.features)
        print_string(f"max len is {self.max_len}")
        

    def _handle_cache(self):
        """ 
            核心是 self.load_data 加载并处理数据，返回原始数据和处理后的特征数据
            需要注意缓存文件与 self.config.cache_dir  self.mode 有关
        Returns:
            samples, features
        """
        os.makedirs(self.config.cache_dir, exist_ok=True)               # 确保缓存文件夹存在
        file_path = getattr(self.config, f'{self.mode}_path').split("/")[-1]
        file = os.path.join(self.config.cache_dir, f"{self.mode}_{file_path}.pt")   # 获得缓存文件的路径   
        if os.path.exists(file) and not self.config.force_reload:       # 如果已经存在，且没有强制重新生成，则从缓存读取
            samples, features = torch.load(file)
            print(f" {len(samples), len(features)} samples, features loaded from {file}")
            return samples, features
        else:
            samples, features = self.load_data()                        # 读取并处理数据
            torch.save((samples, features), file)                       # 生成缓存文件
            return samples, features

    def load_data(self):

        samples, features = [], []
        if self.mode == 'train':
            path = self.config.train_path
        elif self.mode == 'val':
            path = self.config.val_path
        else:
            path = self.config.test_path
        samples = self.datareader.load_samples(path, mode=self.mode)
        features = self.datareader.load_features(samples, self.tokenizer)

        return samples, features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


    def collate_fn(self, batch):
 
        input_ids = torch.tensor(sequence_padding([x.input_ids for x in batch], length=self.max_len,force=True)).long()
        attention_mask = torch.tensor(sequence_padding([x.attention_mask for x in batch], length=self.max_len,force=True)).long()
        token_type_ids = torch.tensor(sequence_padding([x.token_type_ids for x in batch], length=self.max_len,force=True)).long()

        batch = TextBatch(
            input_ids=input_ids, attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        return batch


def get_tokenizer_by_name(config):
    if config.model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained(config.pretrained)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained)
    return tokenizer 

class TextDataModule(pl.LightningDataModule):

    dataset_mapping = {
        'default': TextDataset,
    }
    def __init__(self, config):
        super().__init__()
        self.config = config.data

        self.tokenizer = get_tokenizer_by_name(config)
        print_string("configuration of datamodule")
        print_config(self.config)
        self.DATASET = self.dataset_mapping[self.config.dataset]

    def setup(self, stage=None):
        """
        根据模型运行的阶段，加载对应的数据，并实例化Dataset对象
        """
        if stage == 'fit' or stage is None:
            dataset = self.DATASET(self.config, self.tokenizer, dataset_mode='train')
            if self.config.val_path is None:
                print("number of samples in trainset : ", len(dataset))
                trainsize = int(0.8*len(dataset))
                trainset, valset = random_split(dataset, [trainsize, len(dataset)-trainsize])
                self.trainset, self.valset = trainset, valset 
                self.trainset.collate_fn = self.valset.collate_fn = dataset.collate_fn
                print(f"No val_path provided, split train to {trainsize}, {len(dataset)-trainsize}")
            else:
                self.trainset = dataset
                self.valset = self.DATASET(self.config, self.tokenizer, dataset_mode='val')    
        
        if stage == 'test' or stage is None:
            self.testset = self.DATASET(self.config, self.tokenizer, dataset_mode='test')
            
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.config.train_bsz,
            shuffle=True,
            collate_fn=self.trainset.collate_fn,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.config.val_bsz, 
            shuffle=False,
            collate_fn=self.valset.collate_fn, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.config.test_bsz, 
            shuffle=False,
            collate_fn=self.testset.collate_fn, num_workers=4)

    def predict_dataloader(self):
        return self.test_dataloader()


config = Config.create({
    'model_name':'bert',
    'pretrained' : "/pretrains/pt/chinese-roberta-wwm-ext",
    "data":{
        'dataset': 'default',
        'tokenizer': "/pretrains/pt/chinese-roberta-wwm-ext",
        'train_path': "/home/qing/workspace/datasets/我师兄实在太稳健了.txt",
        'val_path':  None,
        'test_path': None,
        'train_bsz': 4,
        'val_bsz': 4,
        'test_bsz': 4,
        'nways': 8,
        'kshots': 4,
        'cache_dir': './cached/data_utils',
        'force_reload': False
    }
})

def test_batch():
    dm = TextDataModule(config)
    dm.setup('fit')
    print_string("one sample example")
    print(dm.trainset.dataset.samples[0])
    print_string("one feature example")
    print(dm.trainset.dataset.features[0])
    print_string("one batch example")
    for batch in dm.train_dataloader():
        batch.size_info
        print(batch)
        break
        # print(dm.tokenizer.batch_decode(batch.target_ids.cpu().numpy().tolist()))
        input_seq = dm.tokenizer.batch_decode(batch.input_ids.cpu().numpy().tolist())
        if any( "<unk>" in x for x in input_seq ):
            print(input_seq)
            input()
        if torch.any(batch.input_ids == 3):
            print(input_seq)
            input()

if __name__ == '__main__':
    test_batch()