# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 22:31:58 2023

@author: TEJA
"""
print("current status: intializing web app")
import torch
from flask import Flask, render_template, request
from evaluation import translator, infer_sent

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate')
def translate():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    translator.load_state_dict(torch.load("models/translator_model_epoch14.pt",map_location=torch.device('cpu')))
    
        
    translator.to(device)
    translator.eval()
    print("current status: model ready")
    return render_template('translate.html')

@app.route('/process', methods=['POST'])
def process():
    input_text = request.form['text']
    app.logger.info('testing info log')
    translated_text = infer_sent(input_text)
    app.logger.info('input text: ',input_text)
    app.logger.info('translated_text: ',translated_text)
    return translated_text

if __name__ == '__main__':
    app.run()
