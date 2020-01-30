# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 21:52:44 2020

@author: shubh
"""
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import numpy as np
from nltk.tokenize import TreebankWordTokenizer as tbt

import re


def preprocess_text(text):
    text = text.lower()
    
    text = re.sub('\'ll', ' will', text)
    text = re.sub('won\'t', 'will not', text)
    text = re.sub('\'t', ' not', text)
    text = re.sub('I\'m', 'I am', text)

    tokenizer = tbt()
    text = tokenizer.tokenize(text)

    stemmer = SnowballStemmer('english')
    text = [stemmer.stem(word) for word in text]
    
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text if word.isalpha()]
    
    return ' '.join(text)

def cosine_similarity(v1, v2):
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if (not mag1) or (not mag2):
        return 0
    return np.dot(v1, v2) / (mag1 * mag2)

def get_similarity(v1, v2):
    return cosine_similarity(v1, v2)

def get_embeddings(model, list_of_texts):
    cleaned_texts = list(map(preprocess_text, list_of_texts))

    embeddings = model(cleaned_texts)
    embedding_matrix = embeddings.numpy()
    
    return embedding_matrix

def get_top_replies(embedding_matrix, message_embedding):
    score_matrix = np.array([get_similarity(embedding, message_embedding[0]) for embedding in embedding_matrix])
    top_idx = np.argsort(score_matrix)[::-1][:3]
    top_scores = [score_matrix[ind] for ind in top_idx]
    return top_idx, top_scores

def is_bot(user_id, bot_id):
    return user_id == bot_id

def is_public(channel):
    return channel and channel[0] == 'C'

def is_bot_tagged(message, bot_id_regex):
    return re.match(bot_id_regex, message)

def get_clean_message(message, bot_id_regex):
    return re.sub(bot_id_regex, '', message)

def send_reply(user, webclient, channel_id, message):
    if is_public(channel_id):
        message = '<@{}> '.format(user) + message
    
    webclient.chat_postMessage(
      channel = channel_id,
      text = message,
      as_user = True
    )

def send_block(webclient, channel_id, block):
    webclient.chat_postMessage(
      channel = channel_id,
      blocks = block,
      as_user = True
    )

def is_single_word(message):
    return  len(message.split()) == 1

def get_image_data(image_text, image_url, alt_text):
    block = '''[
    	{
    		"type": "image",
    		"title": {
    			"type": "plain_text",
    			"text": "{}",
    			"emoji": true
    		},
    		"image_url": "{}",
    		"alt_text": "{}"
    	}
    ]'''
    return block