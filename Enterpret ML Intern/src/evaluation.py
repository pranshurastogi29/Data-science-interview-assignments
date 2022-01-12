from dataclass import *
from model import *
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sklearn
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
tqdm.pandas()

def infer_sent(df, model):
    df['text'] = df['text'].str.lower()
    df['aspect'] = df['aspect'].str.lower()
    df['text'] = df['text'] + ' Aspect: ' + df['aspect']
    df['label'] = 0
    dataset = DatasetRetriever(
        labels_or_ids=df['label'].values, 
        comment_texts=df['text'].values, 
        lang='en',
        test = True)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        pin_memory=False,
        drop_last=False,
        num_workers=0
    )
    model.eval()
    t = time.time()
    l = []
    device = torch.device("cuda")
    model.to(device)
    for step, (ids, inputs, attention_masks) in enumerate(loader):
        with torch.no_grad():
            inputs = inputs.to(device, dtype=torch.long) 
            attention_masks = attention_masks.to(device, dtype=torch.long)
            outputs = model(inputs, attention_masks)
            toxics = np.argmax(nn.functional.softmax(outputs, dim=1).data.cpu().numpy(), axis = -1)
            l.append(toxics)
    l = np.array(l)
    return l.ravel()


def model_load(path):
    device = torch.device("cuda")
    model_config = transformers.RobertaConfig.from_pretrained('roberta-base')
    model_config.output_hidden_states = True
    model = ABSAModel(conf=model_config)
    model.load_state_dict(torch.load(path))
    return model

def predict_aspect(
    original_text, 
    target_string, 
    idx_start, 
    idx_end, 
    offsets,
    verbose=False):
    
    if idx_end < idx_start:
        idx_end = idx_start
    
    filtered_output  = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_text[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "

    if len(original_text.split()) < 2:
        filtered_output = original_text

    return filtered_output, target_string

def infer_aspect_sent(path, model, device):
    dfx = pd.read_csv(path)
    dfx.columns = ['text','aspect']
    dfx['aspect'] = dfx['aspect'].str.lower()
    dfx['text'] = dfx['text'].str.lower()
    dfx['label'] = 1

    dataset = ABSADataset(
        text=dfx['text'].values,
        sentiment=dfx.label.values,
        selected_text=dfx.aspect.values
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    model.to(device)
    model.eval()
    aspects = []
    sent = []
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Validating")
    with torch.no_grad():
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            orig_selected = d["orig_selected"]
            orig_text = d["orig_text"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"].cpu().numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)
            
            outputs_start, outputs_end, outputs_sent = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            outputs_sent = torch.softmax(outputs_sent, dim=1).cpu().detach().numpy().argmax(axis=1)
            
            for px, text in enumerate(orig_text):
                selected_text = orig_selected[px]                
                filtered_output, target_string = predict_aspect(
                    original_text=text,
                    target_string=selected_text,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                )
                aspects.append(filtered_output)
            sent.append(outputs_sent)
        return aspects, sent

    
test_path = 'pranshu_rastogi_ABSA/data/train_test_data/test.csv'
multi_task_path = 'pranshu_rastogi_ABSA/models/roberta_multi_task_train/model_2.bin'
simple_model_path = 'pranshu_rastogi_ABSA/models/roberta_plain/last-checkpoint9.bin'

model = model_load(multi_task_path)
df = pd.read_csv(test_path)

w, s = infer_aspect_sent(test_path, model, 'cuda:0')
s = np.array(s)
s = s.ravel()
df.columns = ['text','aspect']
df['pred_aspect'] = pd.Series(w)
df['pred_aspect'] = df['pred_aspect'].apply(lambda x: x.strip())

model = SentSimpleNNModel()
model.load_state_dict(torch.load(simple_model_path))

x = infer_sent(df, model = model)
df['label'] = pd.Series(x)

df.to_csv('pranshu_rastogi_ABSA/data/results/test.csv',index = False)
