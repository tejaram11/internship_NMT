# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 16:40:06 2023

@author: TEJA
"""

from vocab.vocab_eng import english_vocabulary, index_to_eng, eng_to_index
from vocab.vocab_tel import telugu_vocabulary, index_to_tel, tel_to_index
from load_from_files import english_sentences, telugu_sentences

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

