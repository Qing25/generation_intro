import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM,  AutoTokenizer, AutoModelForCausalLM, BertTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, BartTokenizer
from tqdm import tqdm
import datasets
import random
from itertools import groupby
import numpy as np 
from omegaconf import OmegaConf
import nlp2 
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint

cfg = {
    'mask_prob': 0.15,
    'poisson_lam': 3,
}
config = OmegaConf.create(cfg)

tokenizer = BartTokenizer.from_pretrained("/pretrains/pt/facebook-bart-base")
MASKTOK = tokenizer.mask_token
model = AutoModelForSeq2SeqLM.from_pretrained("/pretrains/pt/facebook-bart-base")

dataset = datasets.Dataset.from_list([ {'text':line.strip()} for line in open("/home/qing/datasets/qidian/我师兄实在太稳健了.txt") if line.strip() != "" ])



def noisy(examples):
    try:
        target_sent = examples['text']
        sent = examples['text'].split(".")
        random.shuffle(sent)
        input_sent = ".".join(sent)

        sent = input_sent.split("。")
        random.shuffle(sent)
        input_sent = "。".join(sent)

        input_sent = nlp2.split_sentence_to_array(input_sent)
        for ind, word in enumerate(input_sent):
            prob = random.random()
            if prob <= config.mask_prob and len(word) > 0:
                length = np.random.poisson(lam=config.poisson_lam)
                if length == 0:
                    input_sent.insert(ind, MASKTOK)
                else:
                    input_sent[ind:ind + length] = [MASKTOK] * len(input_sent[ind:ind + length])
        input_sent = [k for k, _ in groupby(input_sent)]  # merge_repeat
        input_sent = nlp2.join_words_to_sentence(input_sent)
        examples['input_sent'] = input_sent
        examples['target_sent'] = target_sent
    except Exception as e:
        print(e)
    return examples


p_dataset = dataset.map(noisy, batched=False, num_proc=8)
# p_dataset.save_to_disk(".cached/processed4bart.ds",)
# print(p_dataset)
# p_dataset.to_csv(f'.cached/processed4bart.csv', columns=['input_sent', 'target_sent'], header=False,index=False)

def tokenization_seq2seq(sample):
    td = tokenizer(sample['input_sent'])
    td2 = tokenizer(sample['target_sent'])
    return {
        'input_ids':td.input_ids, 'attention_mask': td.attention_mask,
        'labels': td2.input_ids,
    }

t_dataset = p_dataset.map(tokenization_seq2seq, batched=True, num_proc=4)
t_dataset.set_format(type='torch', columns=["input_ids", "attention_mask", 'labels'])

from q_snippets.data import sequence_padding
class MyCollator:

    def __call__(self, features, return_tensors=None):

        input_ids = torch.tensor(sequence_padding([x['input_ids'].numpy() for x in features], length=None,force=True)).long()
        attention_mask = torch.tensor(sequence_padding([x['attention_mask'].numpy() for x in features], length=None,force=True)).long()
        # token_type_ids = torch.tensor(sequence_padding([x.token_type_ids for x in batch], length=self.max_len,force=True)).long()
        labels = torch.tensor(sequence_padding([x['input_ids'].numpy() for x in features], length=None,force=True)).long()
        decoder_input_ids = torch.tensor(sequence_padding([x['input_ids'].numpy() for x in features], length=None,force=True)).long()


        return {
            'input_ids' : input_ids, 'attention_mask' : attention_mask,
            "labels": labels[:, 1:], 
            'decoder_input_ids':decoder_input_ids[:,:-1]
        }
        
print(t_dataset[:2])
dataloader = DataLoader(t_dataset, collate_fn=MyCollator(), batch_size=32, num_works=4)
for i, batch in enumerate(tqdm(dataloader, total=5)):
    print([(k, v.size()) for k,v in  batch.items() ])
    if i == 5:
        break

class BARTPretrainModel(pl.LightningModule):

    def __init__(self, model_config=None, tokenizer_config=None, lr=3e-4):
        super().__init__()
        if model_config is not None:
            pass 
            # https://huggingface.co/transformers/model_doc/bart.html#transformers.BartForConditionalGeneration
            # self.bart = BartForConditionalGeneration.from_pretrained(model_config)
        self.bart = model 
        self.tokenizer = tokenizer
        self.bart.resize_token_embeddings(self.tokenizer.vocab_size)
        self.config = model_config
        self.lr = lr


    def forward(self, input_text):
        outputs = self.bart(
            **self.tokenizer(input_text, return_tensors='pt')
        )
        return outputs

    def training_step(self, batch, batch_idx):
        loss = self.bart(
            **batch
        )[0]
        if self.global_step == 100:
            exit()
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('dev_loss', loss, prog_bar=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=(self.lr or self.learning_rate))

pl_model = BARTPretrainModel()
trainer = pl.Trainer(gpus=1, check_val_every_n_epoch=1, callbacks=[ ModelCheckpoint(
    monitor='dev_loss', filename='{epoch}-{dev_loss:.2f}', save_last=True, )],
                     default_root_dir='./bart_dpt/',  accelerator='cuda')
trainer.tune(pl_model, train_dataloaders=dataloader)
trainer.fit(pl_model, train_dataloaders=dataloader)