#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:00:04 2020

@author: brie
"""
import torch
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from structs import Corpus

import argparse

# for prior finetuning:
# https://colab.research.google.com/github/interactive-fiction-class/interactive-fiction-class.github.io/blob/master/homeworks/language-model/hw4_transformer.ipynb

parser = argparse.ArgumentParser(description=
                                 'Generating GPT-2 prophecies')
# task
parser.add_argument('--task', type=str, default='atis_intent',
                    help='type of task: pos, ner, srl, sentiment, snips_slot, \
                        snips_intent, atis_slot, atis_intent, chunk, proscons, \
                        objsubj, {pos, ner, srl}_{bc, tc, nw_wsj}')
parser.add_argument('--maxlen', type=int, default=20,
                    help='length of each prediction')
parser.add_argument('--npro', type=int, default=1,
                    help='number of prophecies per input')
parser.add_argument('--temp', type=float, default=1.5,
                    help='temperature of language model')
args = parser.parse_args()
                      
NPRO = args.npro # number of prophecies per partial input
MAXLEN = args.maxlen # len of each prophecy
TASK = args.task # which task
TEMPERATURE = args.temp # temperature for language model's generation

OUTLIERS = 60 # sentences with more than this length are ignored


seq2seq=True
# sentiment is seq2label
if TASK in ['sentiment', 'atis_intent', 'snips_intent', 'objsubj',
                   'proscons', 'posneg', 'sent_negpos']:
    seq2seq = False 
my_device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# choose language model
langmodel = GPT2LMHeadModel.from_pretrained('gpt2').to(my_device)
# choose tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# get data
corpus = Corpus(TASK, seq2seq, max_len=OUTLIERS, P=1.1, no_unk=True) # P>1 to avoid UNKs in training
# tokens after which generation stops
eos_tokens = [tokenizer.encode(x)[0] for x in [".", "!", "?", "..."]] + [198] # last stands for \n
# passed for generator, to be ignored during decoding
pad_token = tokenizer.encode('<|endoftext|>')[0] 

def generate_prophecies(sentences, vocab, langmodel, tokenizer, npro=1, 
                        maxlen=10):
    
    # dict to store prophecies, sent_id:list of lists of prophecies with
    # increasingly longer partial inputs
    prophecies = {}
    id2word = {idx:word for word, idx in vocab.items()}
    
    for s, sentence in sentences.items():
        
        if s%100==0:
            print(s,'...')
        # get sentence in words to use GPT2's tokenizer
        sent = [id2word[idx] for idx in sentence]
        # this list will be of length # word in sentence - 1
        premonitions = []
        
        for i in range(1, len(sent)):
            
            # loop over partial inputs, adding one word each time
            input_ids = torch.tensor([tokenizer.encode(" ".join(sent[:i]))], device=my_device)
            # generate considers size of input in maxlen, so we add size of
            # input to have longer prophecies (otherwise it just returns
            # the input itself with nothing generated)
            premos = langmodel.generate(input_ids=input_ids, 
                                        max_length=input_ids.shape[1]+maxlen, 
                                        do_sample=True, 
                                        num_beams=1, 
                                        num_return_sequences=npro, 
                                        temperature=TEMPERATURE,
                                        eos_token_ids=eos_tokens,
                                        pad_token_id=pad_token)
            if NPRO == 1:
                decoded = [tokenizer.decode(premos[0], 
                                            skip_special_tokens=True)]
            else:
                decoded = [tokenizer.decode(premos[0][k], 
                                skip_special_tokens=True) for k in range(npro)]
            # index of this list correspond to size of partial input - 1
            # position 3 stores prophecies for partial input words 1 to 4
            premonitions.append(decoded)
           
        prophecies[s] = premonitions  
    
    return prophecies

# generate prophecies for test sentences only
sentences = {key:value for key, value in corpus.id2seq.items() 
                                                        if key in corpus.test}
gpt2_prophecies = generate_prophecies(sentences, corpus.word2id, langmodel, 
                                      tokenizer, npro=NPRO, maxlen=MAXLEN)

pickle.dump(gpt2_prophecies, open('prophecies/gpt2Prophecies_'+TASK+'_testset', 'wb'))