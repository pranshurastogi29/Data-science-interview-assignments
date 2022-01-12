# %% [code]
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
import transformers

class SentSimpleNNModel(nn.Module):

    def __init__(self):
        super(SentSimpleNNModel, self).__init__()
        self.backbone = AutoModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(in_features=self.backbone.pooler.dense.out_features,out_features=3)
        
    def forward(self, input_ids, attention_masks):
        seq_x, _= self.backbone(input_ids=input_ids, attention_mask=attention_masks, return_dict=False)
        #apool = torch.mean(seq_x, 1)
        mpool, _ = torch.max(seq_x, 1)
        #x = torch.cat((apool, mpool), 1)
        x = self.dropout(mpool)
        return self.linear(x)

class ABSAModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(ABSAModel, self).__init__(conf)
        self.backbone = AutoModel.from_pretrained('roberta-base', config=conf)
        
        self.d0 = nn.Dropout(0.1)
        self.d1 = nn.Dropout(0.1)
        
        self.l0 = nn.Linear(768 * 2, 2)
        self.l1 = nn.Linear(768 * 2, 3)
        
        torch.nn.init.normal_(self.l0.weight, std=0.02)
        torch.nn.init.normal_(self.l1.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        outputs = self.backbone(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids)
        
        out = torch.stack(outputs[2])
        out = torch.cat((out[-1], out[-2]), dim=-1)
        
        # Head 1
        x = self.d0(out)
        logits = self.l0(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        # Head 2
        y = self.d1(out)
        y = self.l1(y)
        y = y[:,0,:]
        sentiment = y.squeeze(-2)

        return start_logits, end_logits, sentiment