# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 16:13:19 2023

@author: TEJA
"""

print("current status: loading sentence files")

english_file_path = 'data/train.en'
telugu_file_path = 'data/train.te'

with open(english_file_path,'r') as file:
    english_sentences = file.readlines()
with open(telugu_file_path,'r',encoding= 'utf-8') as file:
    telugu_sentences = file.readlines()
 
english_sentences = [sentence.rstrip('\n') for sentence in english_sentences]
telugu_sentences = [sentence.rstrip('\n') for sentence in telugu_sentences]

if __name__ == '__main__':
    print(english_sentences[:5])
    print(telugu_sentences[:5])
    

