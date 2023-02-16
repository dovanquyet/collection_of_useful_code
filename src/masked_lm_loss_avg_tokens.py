import torch
from torch import nn
from transformers import AutoModelForMaskedLM
import numpy as np
import time


class AutoModelForZeroShot(nn.Module):
    def __init__(self, model_name_or_path, tokenizer, use_instancemax=False, mode='fast'):
        super(AutoModelForZeroShot, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.use_instancemax = use_instancemax
        self.mode = mode    
        assert mode in ['fast', 'slow'], '`mode` needs to be either `fast` or `slow`'
        # mode is either 
        # 1. 'fast': input full sentence, take the loss form LM or 
        # 2. 'slow': mask each word, compute the prob, then take their product

        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.sep_id = tokenizer.sep_token_id

    def forward(self, input_ids, attention_mask, e4_bindex, labels=None, **kwargs):
        batch_size, n_branches_x_seq_length = input_ids.shape
        bs, vocab_size = batch_size, self.model.config.vocab_size

        if self.use_instancemax:
            seq_length = self.seq_length
            n_branches = n_branches_x_seq_length//seq_length
            bs = batch_size*n_branches
            input_ids = input_ids.reshape([batch_size*n_branches, seq_length])
            attention_mask = attention_mask.reshape([batch_size*n_branches, seq_length])

        loss_mask = torch.where(input_ids != self.pad_id, 1, 0)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id, reduction='none')
        result = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

        with torch.no_grad():
            if self.mode == 'slow' and self.use_instancemax:
                input_ids = input_ids.reshape([batch_size, n_branches, seq_length])
                attention_mask = attention_mask.reshape([batch_size, n_branches, seq_length])
                nll_loss = torch.zeros(batch_size, n_branches).to(input_ids.device)

                for i in range(batch_size):
                    group_input_ids = input_ids[i]
                    b, e = e4_bindex[i]
                    for j in range(b+1, e):
                        temp = group_input_ids.clone()
                        temp[:, j] = self.mask_id
                        output = self.model(input_ids=temp, attention_mask=attention_mask[i])
                        loss = loss_fn(output.logits[:, j, :], group_input_ids[:, j])    # (n_branched, vocab_size), (n_branched,)
                        nll_loss[i, :] -= loss
                    nll_loss[i, :] = nll_loss[i, :]/(e-b-1)

                result['logits'] = nll_loss
        return result

"""
Usage:
    Load dataset, tokenize it
    Pass to this model
    Get loss as zeroshot classification, to compute auc score or accuracy
"""