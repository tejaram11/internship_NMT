# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:05:00 2023

@author: TEJA
"""

print("current status: preparing model for evalution mode")
import torch

from transformer_main import Transformer
from data.vocab.vocab_tel import telugu_vocabulary,tel_to_index, index_to_tel
from data.vocab.vocab_eng import eng_to_index,START, END, PADDING
from mask import create_masks

d_model = 512
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 200
tel_vocab_size = len(telugu_vocabulary)



#creating a translator instance from Transformer class
translator = Transformer(d_model, 
                          ffn_hidden,
                          num_heads, 
                          drop_prob, 
                          num_layers, 
                          max_sequence_length,
                          tel_vocab_size,
                          eng_to_index,
                          tel_to_index,
                          START, 
                          END, 
                          PADDING)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("currently using device : ",device)
translator.load_state_dict(torch.load("models/translator_model_epoch14.pt"))
translator.to(device)

print("currently using device : ",device)


translator.eval()

print(translator)
def infer_sent(eng_sentence):
    eng_sentence = (eng_sentence,)
    tel_sentence = ("",)
    for word_counter in range(max_sequence_length):
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, tel_sentence)
        predictions = translator(eng_sentence,
                              tel_sentence,
                              encoder_self_attention_mask.to(device), 
                              decoder_self_attention_mask.to(device), 
                              decoder_cross_attention_mask.to(device),
                              enc_start_token=False,
                              enc_end_token=False,
                              dec_start_token=True,
                              dec_end_token=False)
        next_token_prob_distribution = predictions[0][word_counter]
        next_token_index = torch.argmax(next_token_prob_distribution).item()
        next_token = index_to_tel[next_token_index]
        tel_sentence = (tel_sentence[0] + next_token, )
        if next_token == END:
            break
    print("current translation: ",eng_sentence," \n",tel_sentence[0])
    
    return tel_sentence[0]



