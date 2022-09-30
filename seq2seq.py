 
from base64 import encode
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embedding = self._load_embed(config)
        
        self.rnn = nn.LSTM(input_size=config.embed_dim, hidden_size=config.hidden_size, bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)

    def _load_embed(self, config):
        if config.embed_type != 'bert':
            return nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embed_dim, padding_idx=0)
        else:
            embed_layer = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embed_dim, padding_idx=0, _weight=None)
            return 
        
    def forward(self, batch):
        
        embedded = self.dropout(self.embedding(batch.input_ids))
        outputs, (hn,cn) = self.rnn(embedded)
        
        print(outputs.size(), hn.size(), cn.size())
        # outputs (bsz, seqlen, dim*n_layer*n_directions)
        # hn = (bsz, dim*n_layer*n_directions)

        #  正向最后一个hidden和反向第一个hidden 拼接，映射还原纬度
        hidden = torch.tanh(self.fc(hn.permute(1,0,2).reshape(outputs.size(0),-1)))
    
        # （bsz,seqlen,dim*2) (bsz,dim)
        return outputs, hidden

class Attention(nn.Module):
    """
        [decoder_hidden ; encoder_output] -->  (bsz, seqlen)
    """
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        """ (bsz,1,dim) (bsz,seqlen,dim)  """
        
        bsz, seqlen, dim = encoder_outputs.size()
        hidden = hidden.view(bsz,1,-1)
        #repeat decoder hidden state src_len times
        hidden = hidden.repeat(1, seqlen, 1)
        
        #  (bsz, seqlen, dec_dim)             
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        #  (bsz,seqlen,1)
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 

        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embed_dim, padding_idx=0)
        self.proj_linear = nn.Linear(config.embed_dim, config.hidden_size)
        self.att = Attention(config.hidden_size, config.hidden_size)
        
        self.rnn = nn.GRU(config.hidden_size*3, hidden_size=config.hidden_size, bidirectional=False, batch_first=True)
        
        self.lm_linear = nn.Linear(config.hidden_size*4, config.tgt_vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, decoder_input_ids, hidden, encoder_outputs):
        """
            将输入decoder_input_id编码， 计算src上的注意力分数，重新加权

        """
        bsz,srclen,dim = encoder_outputs.size()
        embeded = self.proj_linear(self.embedding(decoder_input_ids)).view(bsz, -1, self.config.hidden_size)                    # (bsz, dim)
        attention_socre_over_src = self.att(hidden, encoder_outputs)  # (bsz,srclen)
        
        # print(embeded.size(), attention_socre_over_src.size(), encoder_outputs.size())

        weighted_src =  attention_socre_over_src.unsqueeze(1) @ encoder_outputs   # (bsz,srclen,dim*2) (bsz,1,srclen)
        
        # （bsz,1,dim*2) (bsz,1,dim)
        rnn_input = torch.cat([weighted_src, embeded], dim=-1)
        
        #  加权的src和当前decoder_input_id的表示拼接，输入rnn走一个时间步
        output, hidden = self.rnn(rnn_input, hidden.view(-1,bsz, self.config.hidden_size))

        assert (output == hidden.permute(1,0,2)).all()    # (bsz,1,dim)
        
        #  (bsz,1,dim)  (bsz,1,dim*2) (bsz,1,dim)  映射至词表大小
        logits = self.lm_linear(torch.cat([output, weighted_src, embeded], dim=-1))

        return logits, hidden


class Seq2seqModel(nn.Module):
    def __init__(self, config, encoder, decoder) -> None:
        super().__init__()
        self.config = config 

        self.encoder = encoder 
        self.decoder = decoder 

    def forward(self, batch):
        bsz,srclen = batch.input_ids.size()
        tgtlen = batch.labels.size(1)

        decoder_outputs = torch.zeros(bsz, tgtlen, self.config.tgt_vocab_size)

        encoder_output, hidden = self.encoder(batch)    # 获取encoder的输出: 输入序列的编码向量
        decoder_input_id = batch.labels[:, 0]           # 目标序列，这里先取第一个 <S>
        for i in range(1, tgtlen):
            output, hidden = self.decoder(decoder_input_id, hidden, encoder_output)
            decoder_outputs[:,i] = output.squeeze()

            teacher_forcing = random.random() < self.config.teacher_forcing_ratio
            pred_id = output.argmax(-1)

            decoder_input_id = pred_id if teacher_forcing else batch.labels[:, i]

        return decoder_outputs   # (bsz,tgtlen, vocab_size)

    def inference(self, batch, max_len=15):
        bsz,srclen = batch.input_ids.size()
        encoder_output, hidden = self.encoder(batch) 
        decoder_input_id = torch.tensor([101])
        decoder_outputs = []
        for i in range(max_len):
            output, hidden = self.decoder(decoder_input_id, hidden, encoder_output)
            decoder_outputs.append(output.squeeze())     #(bsz,vocab)
            decoder_input_id = output.argmax(-1)

            if decoder_input_id.item() == 102:   # eos_id 
                break 
        
        pred_ids = torch.stack(decoder_outputs, dim=1)
        return


def train(model, dataloader, optimizer, loss_fn, clip_val=1.0):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(batch)  # (bsz, tgtlen, vocab_size)
        bsz, tgtlen, vocab_size = output.size()

        logits = output[:,1:].view(-1, vocab_size)
        loss = loss_fn(logits, batch.labels[:,1:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)



from dataclasses import dataclass 

@dataclass
class Batch:
    input_ids : torch.Tensor 
    labels : torch.Tensor
    attention_mask : torch.Tensor = None     
    
    
batch = Batch(torch.randint(0,1000,(4,30)), torch.randint(0,1000,(4,20)))

if __name__ == '__main__':
    from omegaconf import OmegaConf
    config_dict = {
        'embed_type' : 'rand', # bert
        'embed_dim' : 768,
        'hidden_size' : 256,
        'dropout' : 0.1,
        'vocab_size' : 1000,
        'tgt_vocab_size': 1000,
        'teacher_forcing_ratio': 0.5
    }
    config = OmegaConf.create(config_dict)

    loss_fn = F.cross_entropy

    encoder = Encoder(config)
    decoder = Decoder(config)
    seq2seq = Seq2seqModel(config, encoder, decoder)

    decoder_output = seq2seq(batch)
    print(seq2seq)
    print(decoder_output.size())

    

def fake_inputs():
    from collections import namedtuple
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("/pretrains/pt/bert-base-cased/")
    sentence = ""
    td = tokenizer(sentence, return_tensors='pt')
    return td 