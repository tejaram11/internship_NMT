# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 16:15:20 2023

@author: TEJA
"""

print("current status: loading vocabulary")

START = ''
PADDING = ''
END = ''


english_vocabulary = [START, '',' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '“', '”', '‘','’', 
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', '@', 
                        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                        'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
                        'Y', 'Z',
                        '[', 
                        '\\', 
                        ']', '^', '_', 
                        '`', 
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                        'y', 'z', 
                        '{', '|', '}', '~', PADDING, END]


index_to_eng = {k:v for k,v in enumerate(english_vocabulary)}
eng_to_index = {v:k for k,v in enumerate(english_vocabulary)}