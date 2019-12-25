import math
import sys
import pandas as pd
import numpy as np
import random as rn
import torch
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertTokenizer
from torch import nn

################################################################################
################################################################################

def square_rooted(x):
    return math.sqrt(sum([a*a for a in x]))
def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return numerator/float(denominator)
################################################################################
################################################################################

def embedTweets(documents,bert,tokenizer):
    
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]'], documents))
    train_tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, train_tokens)), maxlen=128, truncating="post", padding="post", dtype="int")
    

    train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
    
    #print("!Loading Bert Model!")
    #bert = BertModel.from_pretrained('bert-base-uncased')
    #print("! Model Loaded !")
    
    #Convert train token into tensors
    x = torch.tensor(train_tokens_ids)
    _, pooled_output = bert(x, attention_mask=None, output_all_encoded_layers=False)
    
    # return only numpy
    return pooled_output.cpu().detach().numpy()
    
    
