# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:25:33 2023

@author: TEJA
"""

print("current status: training the model")
import signal,sys
import torch
from torch import nn
import time, pickle
from torch.utils.data import DataLoader

from data.vocab.vocab_tel import telugu_vocabulary,tel_to_index, index_to_tel
from data.vocab.vocab_eng import eng_to_index,START, END, PADDING
from data.dataset_custom import dataset
from transformer_main import Transformer
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



optim = torch.optim.Adam(translator.parameters(),lr=1e-4)
criterian = nn.CrossEntropyLoss(ignore_index=tel_to_index[PADDING], reduction='none')

for params in translator.parameters():
    if params.dim()>1:
        nn.init.xavier_uniform_(params)
        
translator.load_state_dict(torch.load('models/interrupted_model_except.pt'))
#optim.load_state_dict(torch.load('models/optimizer_state.pt'))
translator.train()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
translator.to(device)
total_loss = 0
num_epochs = 10
batch_size=105

#to save training data
train_data=dict()
train_losses=[]
train_time=[]

#loading the dataset
train_loader=DataLoader(dataset,batch_size)


def handle_interrupt(signal, frame):
    print("Training interrupted. Saving model...")
    torch.save(translator.state_dict(), 'models/interrupted_model.pt')
    torch.save(optim.state_dict(),'models/optimizer_state.pt')
    print(batch_num,epoch)
    sys.exit(0)

# Register the Ctrl+C signal handler
signal.signal(signal.SIGINT, handle_interrupt)

#training loop
try:
 for epoch in range(5,num_epochs):
    start_time = time.time()
    print(f"Epoch {epoch}")
    iterator= iter(train_loader)
    #batch wise training
    for batch_num, batch in enumerate(iterator):
        #transformer.train()
        eng_batch, tel_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, tel_batch)
        optim.zero_grad()
        tel_predictions = translator(eng_batch,
                                     tel_batch,
                                     encoder_self_attention_mask.to(device), 
                                     decoder_self_attention_mask.to(device), 
                                     decoder_cross_attention_mask.to(device),
                                     enc_start_token=False,
                                     enc_end_token=False,
                                     dec_start_token=True,
                                     dec_end_token=True)
        labels = translator.decoder.sentence_embedding.batch_tokenize(tel_batch, start_token=False, end_token=True)
        loss = criterian(
            tel_predictions.view(-1, tel_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)
        valid_indicies = torch.where(labels.view(-1) == tel_to_index[PADDING], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        train_losses.append(loss.item())
        if batch_num % 10 == 0:
            #print(f"Accuracy: {accuracy}")
            print(f"epoch: {epoch}\tbatch: {batch_num}\tloss: {loss.item()}" )
            print(f"English: {eng_batch[0]}")
            print(f"telugu Translation: {tel_batch[0]}")
            tel_sentence_predicted = torch.argmax(tel_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in tel_sentence_predicted:
                if idx == tel_to_index[END]:
                    break
                predicted_sentence += index_to_tel[idx.item()]
            print(f"telugu Prediction: {predicted_sentence}")
            print("-"*60)
            
            
            '''
            transformer.eval()
            tel_sentence = ("",)
            eng_sentence = ("should we go to the mall?",)
            for word_counter in range(max_sequence_length):
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, tel_sentence)
                predictions = transformer(eng_sentence,
                                          tel_sentence,
                                          encoder_self_attention_mask.to(device), 
                                          decoder_self_attention_mask.to(device), 
                                          decoder_cross_attention_mask.to(device),
                                          enc_start_token=False,
                                          enc_end_token=False,
                                          dec_start_token=True,
                                          dec_end_token=False)
                
                next_token_prob_distribution = predictions[0][word_counter] # not actual probs
                next_token_index = torch.argmax(next_token_prob_distribution).item()
                next_token = index_to_tel[next_token_index]
                tel_sentence = (tel_sentence[0] + next_token, )
                if next_token == END:
                    break
            
            print(f"Evaluation translation (should we go to the mall?) : {tel_sentence}")
            print("-------------------------------------------")
            
            '''
            
    torch.save(translator.state_dict(), f'models/translator_model_epoch{epoch}.pt')        
    end_time=time.time()
    epoch_duration=end_time-start_time
    hours = int(epoch_duration // 3600)
    minutes = int((epoch_duration % 3600) // 60)
    seconds = int(epoch_duration % 60)
    train_data[epoch]={
        'start loss': train_losses[0],
        'final loss': train_losses[-1],
        'avg loss' :  sum(train_losses)/len(train_losses),
        'time taken': (hours,minutes,seconds)
        }
    
except:
    print("Training excepted. Saving model...")
    torch.save(translator.state_dict(), 'models/interrupted_model_except.pt')
    torch.save(optim.state_dict(),'models/optimizer_state.pt')
    print(batch_num,epoch)
    with open(f'models/train_data_{epoch}.pkl', 'wb') as file:
        pickle.dump(train_data, file)
    sys.exit(0)
    
    


#saving the model
torch.save(translator.state_dict(), 'models/translator_model.pt')

print(train_data)

# Save the dictionary using Pickle
with open('models/train_data.pkl', 'wb') as file:
    pickle.dump(train_data, file)
    


    

