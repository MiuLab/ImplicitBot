import pandas as pd

data = pd.read_csv('Description.csv')
sentences = list(data['High-level Description'])

import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(
            self, 
            queries,
            decode_method="beam", 
            num_generate=5, 
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    temperature=1.0,
                    do_sample=False,
                    top_p=0.9,
                    top_k=40,
                    repetition_penalty=1.0,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=10,
                    num_return_sequences=10,
                    )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)

            return decs
        
for i in range(len(sentences)):
    s = sentences[i].strip()
    if s[-1] not in ['.', '?', '!']:
        s += '.'
    sentences[i] = s
    
print("model loading ...")
comet = Comet("./comet-atomic_2020_BART")
comet.model.zero_grad()
print("model loaded")
queries = []
relations = ["xIntent", "xNeed", "xWant", "isAfter", "isBefore"]
dic = {}
for relation in relations:
    dic[relation] = []
    queries = []
    for sentence in sentences:
        query = "{} {} [GEN]".format(sentence, relation)
        queries.append(query)
    results = comet.generate(queries, decode_method="beam", num_generate=5)
    for i, result in enumerate(results):
        final_result = []
        j = 0
        while len(final_result) < 2 and j<5:
            if result[j] != ' none':
                final_result.append(result[j])
            j += 1
        dic[relation].append(final_result)
#         print(rs[i], r)
# print(queries)

import torch
from transformers import GPTJForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16)
# model.to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

from parallelformers import parallelize

parallelize(model, num_gpus=2, fp16=True, verbose='detail')

total_apps = []

apps = []
prefix1 = "PersonX intends"
for sen in dic['xIntent']:
    p = prefix1 + sen[0].replace('.','') + ' and' + sen[1].replace('.','') + prefix
    
    inputs = tokenizer([p], return_tensors="pt")
    outputs = model.generate(
        **inputs,
#         num_beams=5,
#         no_repeat_ngram_size=4,
        max_length=50,
        do_sample=True, 
        temperature=0.01,
        top_p=0.9,
    )

    outputs = tokenizer.batch_decode(outputs)
    app = outputs[0].split('.')[0].replace(p+' ', '')
    apps.append(app)
    
for i in range(len(apps)):
    apps[i] = apps[i].replace('”', '').replace('“', '').replace('"', '')
total_apps.append(apps)

apps = []
prefix1 = "PersonX needs"
for sen in dic['xNeed']:
    p = prefix1 + sen[0].replace('.','') + ' and' + sen[1].replace('.','') + prefix
    
    inputs = tokenizer([p], return_tensors="pt")
    outputs = model.generate(
        **inputs,
#         num_beams=5,
#         no_repeat_ngram_size=4,
        max_length=50,
        do_sample=True, 
        temperature=0.01,
        top_p=0.9,
    )

    outputs = tokenizer.batch_decode(outputs)
    app = outputs[0].split('.')[0].replace(p+' ', '')
    apps.append(app)
    
for i in range(len(apps)):
    apps[i] = apps[i].replace('”', '').replace('“', '').replace('"', '')
total_apps.append(apps)

apps = []
prefix1 = "PersonX wants"
for sen in dic['xWant']:
    p = prefix1 + sen[0].replace('.','') + ' and' + sen[1].replace('.','') + prefix
    
    inputs = tokenizer([p], return_tensors="pt")
    outputs = model.generate(
        **inputs,
#         num_beams=5,
#         no_repeat_ngram_size=4,
        max_length=50,
        do_sample=True, 
        temperature=0.01,
        top_p=0.9,
    )

    outputs = tokenizer.batch_decode(outputs)
    app = outputs[0].split('.')[0].replace(p+' ', '')
    apps.append(app)
    
for i in range(len(apps)):
    apps[i] = apps[i].replace('”', '').replace('“', '').replace('"', '')
total_apps.append(apps)

apps = []
prefix1 = ""
for sen in dic['isAfter']:
    if len(sen) == 0:
        apps.append('')
    else:
        p = prefix1 + sen[0].replace('.','') + ' and' + sen[1].replace('.','') + prefix

        inputs = tokenizer([p], return_tensors="pt")
        outputs = model.generate(
            **inputs,
    #         num_beams=5,
    #         no_repeat_ngram_size=4,
            max_length=50,
            do_sample=True, 
            temperature=0.01,
            top_p=0.9,
        )

        outputs = tokenizer.batch_decode(outputs)
        app = outputs[0].split('.')[0].replace(p+' ', '')
        apps.append(app)
    
for i in range(len(apps)):
    apps[i] = apps[i].replace('”', '').replace('“', '').replace('"', '')
total_apps.append(apps)

apps = []
prefix1 = ""
for sen in dic['isBefore']:
    if len(sen) == 0:
        apps.append('')
    else:
        p = prefix1 + sen[0].replace('.','') + ' and' + sen[1].replace('.','') + prefix

        inputs = tokenizer([p], return_tensors="pt")
        outputs = model.generate(
            **inputs,
    #         num_beams=5,
    #         no_repeat_ngram_size=4,
            max_length=50,
            do_sample=True, 
            temperature=0.01,
            top_p=0.9,
        )

        outputs = tokenizer.batch_decode(outputs)
        app = outputs[0].split('.')[0].replace(p+' ', '')
        apps.append(app)
    
for i in range(len(apps)):
    apps[i] = apps[i].replace('”', '').replace('“', '').replace('"', '')
total_apps.append(apps)

import numpy as np
total_apps = np.array(total_apps).T.tolist()
for i in range(total_apps):
    total_apps[i] = list(set(total_apps[i]))

data['predict app sequence'] = total_apps


app2cate = pd.read_csv('App_to_Category.csv')
app2cate = app2cate.set_index('Play Store Name').T.to_dict('list')
for app in app2cate:
    app2cate[app.lower()] = app2cate[app]

categories = []
for apps in total_apps:
    category = []
    for app in apps:
        if app.lower() in app2cate.keys():
            category.append(app2cate[app.lower()][0])
    categories.append(category)
    
data['predict app category'] = categories
data.to_csv('predict result.csv')