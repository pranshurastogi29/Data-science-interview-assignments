# %% [code]
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import time
import random
from datetime import datetime
from tqdm import tqdm
tqdm.pandas()
import transformers
import gc
import re
from transformers import AutoTokenizer, AutoModel

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

def process_data(text, selected_text, sentiment, tokenizer, max_len):
    text = " " + " ".join(str(text).split())
    selected_text = " " + " ".join(str(selected_text).split())

    len_st = len(selected_text) - 1
    idx0 = None
    idx1 = None

    for ind in (i for i, e in enumerate(text) if e == selected_text[1]):
        if " " + text[ind: ind+len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    char_targets = [0] * len(text)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1
    
    tok_text = tokenizer.encode_plus(text, add_special_tokens=True, 
            max_length=max_len, 
            pad_to_max_length=True,
            return_offsets_mapping = True,
            return_token_type_ids = True)
    
    text_offsets = tok_text['offset_mapping']
    
    target_idx = []
    for j, (offset1, offset2) in enumerate(text_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)
    
    targets_start = target_idx[0]
    targets_end = target_idx[-1]    
    input_ids = tok_text['input_ids']
    token_type_ids = tok_text['token_type_ids']
    mask = tok_text['attention_mask']
    text_offsets = text_offsets

    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'orig_text': text,
        'orig_selected': selected_text,
        'sentiment': int(sentiment),
        'offsets': text_offsets
        }
        
class DatasetRetriever(Dataset):

    def __init__(self, labels_or_ids, comment_texts, lang = 'en', test=False):
        self.test = test
        self.lang = lang
        self.labels_or_ids = labels_or_ids
        self.comment_texts = comment_texts
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base",lowercase=True)
    def get_tokens(self, text):
        encoded = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=256, 
            pad_to_max_length=True
        )
        return encoded['input_ids'], encoded['attention_mask']

    def __len__(self):
        return self.comment_texts.shape[0]

    def __getitem__(self, idx):
        text = self.comment_texts[idx]
        if self.test is False:
            label = self.labels_or_ids[idx]
            target = onehot(3, label)

        tokens, attention_mask = self.get_tokens(str(text))
        tokens, attention_mask = torch.tensor(tokens), torch.tensor(attention_mask)

        if self.test is False:
            return target, tokens, attention_mask
        return self.labels_or_ids[idx], tokens, attention_mask

    def get_labels(self):
        return list(np.char.add(self.labels_or_ids.astype(str),''))

class ABSADataset:
    def __init__(self, text, sentiment, selected_text):
        self.text = text
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base",lowercase=True)
        self.max_len = 256
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        data = process_data(
            self.text[item], 
            self.selected_text[item],
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )
        
        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_text': data["orig_text"],
            'orig_selected': data["orig_selected"],
            'sentiment': torch.tensor(data['sentiment'], dtype=torch.long),
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }