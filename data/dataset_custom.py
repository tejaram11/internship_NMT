# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 20:09:41 2023

@author: TEJA
"""
print("current status: loading dataset")
from torch.utils.data import Dataset
from data.vocab.vocab_eng import english_vocabulary
from data.vocab.vocab_tel import telugu_vocabulary
from data.load_from_files import english_sentences, telugu_sentences



max_seq_length = 200

def is_valid_tokens(sentence,vocab):
    for token in list(set(sentence)):
        
        if token not in vocab:
            return False
    return True

def is_valid_length(sentence,max_seq_length):
    return len(list(sentence)) < (max_seq_length-1)

valid_sentence_indices = []

for index in range(len(telugu_sentences)):
    telugu_sentence, english_sentence = telugu_sentences[index], english_sentences[index]
    if is_valid_length(telugu_sentence,max_seq_length) \
    and is_valid_length(english_sentence,max_seq_length) \
    and is_valid_tokens(telugu_sentence,telugu_vocabulary) \
    and is_valid_tokens(english_sentence,english_vocabulary):
        valid_sentence_indices.append(index)


        
telugu_sentences_processed = [telugu_sentences[i] for i in valid_sentence_indices]
english_sentences_processed = [english_sentences[i] for i in valid_sentence_indices]

telugu_sentences_processed = telugu_sentences_processed[:2560000]
english_sentences_processed = english_sentences_processed[:2560000]


class TextDataSet(Dataset):
    def __init__(self, source, target):
        super().__init__
        self.english_sentences = source
        self.telugu_sentences = target
        
    def __len__(self):
        return len(self.english_sentences)
    
    def __getitem__(self,idx):
        return self.english_sentences[idx], self.telugu_sentences[idx]
    
    
dataset = TextDataSet(english_sentences_processed,telugu_sentences_processed) 
 




