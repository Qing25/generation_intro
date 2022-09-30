
import os,sys 
from typing import Union, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import BartModel, BartTokenizer,BartConfig, BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.utils import logging

logger = logging.get_logger(__name__)

sys.path.append(".")

from model_utils.copy_generation_utils import CopyMechanismMixIn

class BartConditionalGenerationWithCopy(BartForConditionalGeneration, CopyMechanismMixIn):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.copy_linear = nn.Sequential(
            nn.Linear(config.d_model, config.d_model//2),
            nn.ReLU(),
            nn.Linear(config.d_model//2, 1)
        )

        # Initialize weights and apply final processing
        self.post_init()

    def apply_copy_mechanism(
        self, 
        next_token_logits, 
        cross_attentions,
        last_hidden_states, 
        encoder_input_ids
    ):
        """ 
            next_token_logits: lm_logits (bsz, tgt, vocab_size)
            cross_attentions : 6 layers of (bsz,nheads,1,src_len)
            decoder_hidden_states: 7 layers of (bsz, 1, dim)
            encoder_input_ids : (bsz, src_len)
        """
        if type(last_hidden_states) is not torch.Tensor:
            last_hidden_states = last_hidden_states[-1]

        # 1. 计算拷贝的概率            
        p_copy = torch.sigmoid(self.copy_linear(last_hidden_states))   # (bsz,tgt, 1)

        # 2. 将模型预测的词的概率乘以 1-p_copy
        pre_word_prob = torch.softmax(next_token_logits.view(*p_copy.size()[:2], -1), dim=-1) * (1 - p_copy)
        # 3. 输入序列单词的概率
        encoder_word_attention = p_copy * torch.mean(cross_attentions[-1], dim=1)  # (bsz,12,tgtlen,srclen) - > (bsz,tgtlen,srclen)

        bsz,tgt_len,src_len = encoder_word_attention.size()
        # 4. padding的位置masked掉
        mask = torch.where(encoder_input_ids == 1,
                        encoder_word_attention.new_zeros(encoder_input_ids.shape),
                        encoder_word_attention.new_ones(encoder_input_ids.shape))
        encoder_word_attention = encoder_word_attention * mask.unsqueeze(1)

        personal_words = encoder_input_ids.unsqueeze(1).repeat(1, tgt_len, 1)  # (bsz,src_len) -> (bsz,tgt,src)
                            #     (bsz,tgt, vsize)     (bsz,tgt,src)    (bsz,tgt,src)
        # 5.把两部分概率加起来
        word_prob = torch.scatter_add(pre_word_prob, 2, personal_words, encoder_word_attention)
        
        return word_prob

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        assert output_attentions is True
        assert output_hidden_states is True 
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        def prob_ce(pred_prob, labels):
            """  (bsz,tgt,vsize) - (bsz,tgt)  进来的已经是两个概率分布的和了，所以取对数之后算nll loss
            """
            pred_prob = torch.clamp(pred_prob, min=1e-12)
            return F.nll_loss(torch.log(pred_prob.view(-1, pred_prob.size(-1))), labels.view(-1), ignore_index=-100)
        
        final_prob = None 
        masked_lm_loss = None
        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            final_prob = self.apply_copy_mechanism(
                lm_logits, outputs.cross_attentions, outputs.last_hidden_states, input_ids
            )
            masked_lm_loss = prob_ce(final_prob, labels)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            # logits=lm_logits,
            logits=lm_logits if final_prob is None else final_prob,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )



def _test():
    tokenizer = BartTokenizer.from_pretrained("/pretrains/pt/facebook-bart-base")
    model = BartConditionalGenerationWithCopy.from_pretrained("/pretrains/pt/facebook-bart-base")
    print(model)

    td = tokenizer("You know nothing, but ", return_tensors='pt')
    pred_ids = model.generate(
        inputs=td.input_ids, enable_copy=True, output_attentions=True, output_hidden_states=True,
        max_length=30
    )
    print(tokenizer.batch_decode(pred_ids))


if __name__ == '__main__':
    
    _test()