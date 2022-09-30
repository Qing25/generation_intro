from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
import math
import os
from pathlib import Path

import datasets as ds
import pytorch_lightning as pl
import transformers as tr
import torch
from torch.utils.data import DataLoader

from typing import *
from io import BytesIO

import dill
import xxhash

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import logging

logger = logging.getLogger()
CACHE_DIR = ".cached"

def fingerprint_dict(state: Dict[str, Any]):
    m = xxhash.xxh64()
    for key in sorted(state):
        m.update(key)
        m.update(dumps(state[key]))
    return m.hexdigest()

def dumps(obj: Any):
    """pickle an object to a string"""
    file = BytesIO()
    dill.Pickler(file, recurse=True).dump(obj)
    return file.getvalue()

@dataclass
class GPTDataModuleConfig:
    dataset_name: List[str] = field(
        default=None,
        metadata={'help': 'Huggingface dataset name.'}
    )
    dataset_config_name: List[str] = field(
        default=None,
        metadata={'help': 'Huggingface dataset config name.'}
    )
    valid_split: float = field(
        default=0.05,
        metadata={'help': 'Fraction of dataset to reserve for validation.'}
    )
    text_col: str = field(
        default='text',
        metadata={'help': 'Name of text column in Huggingface dataset.'}
    )
    hf_tokenizer: str = field(
        default='gpt2',
        metadata={'help': 'Name of pretrained Huggingface tokenizer.'}
    )
    dataset_seq_len: int = field(
        default=1024,
        metadata={'help': 'Sequence length (in tokens) of examples in dataset. '
                          'Note: this acts as an upper bound on sequence length '
                          'during training, which can be adjusted to fractions '
                          'of this value.'}
    )
    max_seq_len: int = field(
        default=1024,
        metadata={'help': 'Sequence length of examples during training.'}
    )
    per_device_bsz: int = field(
        default=256,
        metadata={'help': 'Batch size (per device).'}
    )
    num_preprocess_workers: int = field(
        default=4,
        metadata={'help': 'Number of workers for dataset preprocessing.'}
    )
    num_dataloader_workers: int = field(
        default=0,
        metadata={'help': 'Number of workers for dataloader.'}
    )
    data_seed: int = field(
        default=1234,
        metadata={'help': 'Seed for dataset splitting and masking.'}
    )
    cache_dir: str = field(
        default=CACHE_DIR,
        metadata={'help': f"Directory for caching preprocessed dataset.\n"
                          f"Defaults to {CACHE_DIR}"}
    )
    overwrite: bool = field(
        default=False,
        metadata={'help': "Rerun preprocessing and overwrite cache."}
    )
    shuffle: bool = field(
        default=True,
        metadata={'help': "Shuffle dataset."}
    )
    shuffle_seed: int = field(
        default=0xBE575E3D,
        metadata={'help': "Seed for shuffling dataset."}
    )

    def __post_init__(self):
        self.dataset_name = self.dataset_name or ['wikipedia', 'bookcorpusopen']
        self.dataset_config_name = self.dataset_config_name or ['20220301.en', 'plain_text']
        self.fingerprint = self._init_fingerprint()

        # bsz/seqlen rescaling
        bsz_rescale = int(self.dataset_seq_len / self.max_seq_len)
        if bsz_rescale * self.max_seq_len != self.dataset_seq_len:
            raise ValueError(f"max_seq_len ({self.max_seq_len}) must factorize "
                             f"dataset_seq_len ({self.dataset_seq_len})")
        self.dataloader_bsz = int(self.per_device_bsz / bsz_rescale)
        if bsz_rescale * self.dataloader_bsz != self.per_device_bsz:
            raise ValueError(f"per_device_bsz ({self.per_device_bsz}) must "
                             f"be divisible by the ratio of dataset_seq_len "
                             f"({self.dataset_seq_len}) to max_seq_len "
                             f"({self.max_seq_len}).")

    def _init_fingerprint(self):
        KEYS_TO_HASH = [
            'dataset_name',
            'dataset_config_name',
            'valid_split',
            'text_col',
            'hf_tokenizer',
            'dataset_seq_len'
        ]
        state = self.__dict__
        state = {k: state[k] for k in KEYS_TO_HASH}
        return fingerprint_dict(state)


class GPTDataModule(pl.LightningDataModule):
    def __init__(self, config: GPTDataModuleConfig):
        super().__init__()
        self.config = config
        self.cache_file_path = self._get_cache_file_path()
        self.tokenizer = tr.AutoTokenizer.from_pretrained(config.hf_tokenizer)

    def _get_cache_file_path(self):
        fp = self.config.fingerprint
        return os.path.join(self.config.cache_dir, fp)

    def prepare_data(self) -> None:
        if Path(self.cache_file_path).exists() and not self.config.overwrite:
            logger.info(f"Preprocessed dataset already cached in "
                        f"{self.cache_file_path}, skipping `prepare_data`.")
            return

        raw_datasets = self._load_raw_datasets()
        dataset_splits = defaultdict(list)
        dataset_dict = ds.DatasetDict()

        for k,d in raw_datasets.items():
            logger.info(f"Preprocessing dataset: {k}")
            d = self._preprocess_raw_datasets(d)
            d = self._split_raw_datasets(d)
            for split in d.keys():
                dataset_splits[split].append(d[split])

        for split, subsets in dataset_splits.items():
            dataset_dict[split] = ds.concatenate_datasets(subsets)

        logger.info(f"Caching processed dataset to {self.cache_file_path}")
        dataset_dict.save_to_disk(self.cache_file_path)

    def _load_raw_datasets(self):
        c = self.config
        raw_datasets = {}
        for name, config in zip(c.dataset_name, c.dataset_config_name):
            raw_dataset_key = f"{name}_{config}"
            raw_datasets[raw_dataset_key] = ds.load_dataset(name, config)
        return raw_datasets

    def _preprocess_raw_datasets(self, d: ds.DatasetDict, bsz: int = 128):
        c = self.config
        block_size = c.dataset_seq_len

        def batch_preproc_function(examples):
            examples = self.tokenizer(
                examples[c.text_col],
                add_special_tokens=False,
                return_attention_mask=False
            )
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        preproc_datasets = d.map(
            batch_preproc_function,
            batched=True,
            batch_size=bsz,
            writer_batch_size=bsz,
            num_proc=c.num_preprocess_workers,
            remove_columns=d['train'].column_names,
            desc="Running tokenizer on dataset",
        )

        return preproc_datasets

    def _split_raw_datasets(self, d: ds.DatasetDict) -> ds.DatasetDict:
        if not self.config.valid_split:
            logger.info("No validation split specified, skipping validation split")
        elif "validation" in d:
            logger.info("Validation set already in raw datasets, skipping validation split")
        elif "train" in d:
            split_dataset = d["train"].train_test_split(
                test_size=self.config.valid_split,
                seed=self.config.data_seed,
            )
            d["train"] = split_dataset["train"]
            d["validation"] = split_dataset["test"]
        else:
            logger.info("Train set not in raw datasets, skipping validation split")
        return d

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = ds.load_from_disk(self.cache_file_path)
        if self.config.shuffle:
            self.dataset['train'] = self.dataset['train'].shuffle(seed=self.config.shuffle_seed)

    def collate_fn(self, examples: List[Dict[str, Any]]):
        batch = {
            'input_ids': torch.tensor([x['input_ids'] for x in examples]).view(
                self.config.per_device_bsz, self.config.max_seq_len
            )
        }
        batch['labels'] = batch['input_ids'].clone()
        return batch

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        c = self.config
        return DataLoader(
            self.dataset['train'],
            batch_size=c.dataloader_bsz,
            collate_fn=self.collate_fn,
            num_workers=c.num_dataloader_workers,
            shuffle=False,
            drop_last=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        c = self.config
        return DataLoader(
            self.dataset['validation'],
            batch_size=c.dataloader_bsz,
            collate_fn=self.collate_fn,
            num_workers=c.num_dataloader_workers,
            shuffle=False,
            drop_last=True,
        )


if __name__ == '__main__':
    c = GPTDataModuleConfig(
        dataset_name=['wikipedia'],
        dataset_config_name=['20220301.en'],
        num_preprocess_workers=8,
        max_seq_len=512,
    )
    dm = GPTDataModule(c)
    dm.prepare_data()
    dm.setup()
    print(len(dm.train_dataloader()))