"""
https://github.com/mirandrom/lightning-transformer-pretraining
"""


from dataclasses import dataclass, field
import math

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
import transformers as tr

from typing import *

class GPTModel(pl.LightningModule):
    def __init__(self, config: GPTModelConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.__dict__)
        pl.seed_everything(self.config.model_seed)
        self.model = self.init_model()
        self.compcurr_len_gen = self.compcurr_len_sampler()

    def init_model(self):
        c = self.config
        model_config = tr.GPT2Config(
            n_embd=c.n_embd,
            n_layer=c.n_layer,
            n_head=c.n_head,
            n_inner=c.n_inner,
        )
        pl.seed_everything(c.model_seed)
        model = tr.AutoModelForCausalLM.from_config(model_config)
        return model

    def get_learning_rate(self):
        # (equation D.1, page 26 https://arxiv.org/pdf/2001.08361.pdf)
        n = self.model.num_parameters(exclude_embeddings=True)
        lr = 0.003239 - 0.0001395 * math.log(n)
        return lr

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        lr = self.config.learning_rate or self.get_learning_rate()
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.adam_wd,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=lr,
                          betas=(self.config.adam_beta_1,
                                 self.config.adam_beta_2),
                          eps=self.config.adam_eps
                          )
        lr_scheduler = tr.get_scheduler(
            name=self.config.lr_scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=self.config.num_warmup_total_steps,
        )
        lr_dict = dict(
            scheduler=lr_scheduler,
            interval="step",
        )
        return dict(optimizer=optimizer, lr_scheduler=lr_dict)

    def forward(self, batch):
        out = self.model(batch['input_ids'],
                         labels=batch['labels'])
        return out

    def compcurr_len_sampler(self):
        while True:
            samples = torch.multinomial(self.config.compcurr_weights,
                                        num_samples=self.config.compcurr_sampler_bsz,
                                        replacement=True)
            for s in samples:
                yield self.config.compcurr_lens[s]

    def training_step(self, batch, batch_idx):
        if self.config.compcurr:
            input_ids = batch['input_ids']
            device = input_ids.device
            position_ids = torch.arange(input_ids.shape[-1], dtype=torch.long, device=device)
            position_ids.unsqueeze(0).expand_as(input_ids)
            compcurr_len = next(self.compcurr_len_gen)
            batch['input_ids'] = batch['input_ids'].view(-1, compcurr_len)
            batch['labels'] = batch['labels'].view(-1, compcurr_len)
            batch['position_ids'] = position_ids.view(-1, compcurr_len)
        out = self.forward(batch)
        loss = out.loss
        self.log('train_loss', loss.item())
        # manually log global_step to prevent issues with resuming from ckpt
        # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/13163
        self.log('global_step', float(self.global_step))
        return loss

    def validation_step(self, batch, batch_idx):
        d = {}
        loss = 0
        for eval_len in self.config.compcurr_eval_lens:
            batch['input_ids'] = batch['input_ids'].view(-1, eval_len)
            batch['labels'] = batch['labels'].view(-1, eval_len)
            out = self.forward(batch)
            loss = out.loss
            d[f'eval_loss_{eval_len}'] = loss.item()
        # manually log global_step to prevent issues with resuming from ckpt
        # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/13163
        d['global_step'] = self.global_step
        self.log_dict(d)
        return 