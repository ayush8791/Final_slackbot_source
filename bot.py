# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:48:30 2020

@author: Shubhajit_Chatterjee and Ayush Garg
"""
# Import Libraries
import pandas as pd
import util as ut

import tensorflow as tf
from slack import RTMClient

# Set Parameters
BOT_ID = 'URXE7TQPP'
BOT_TAGGED_PREFIX = '<@{}>'.format(BOT_ID)
BOT_ID_REGEX = '^{}'.format(BOT_TAGGED_PREFIX)
SLACK_TOKEN = ''
SYLLABUS=""" 
*S.No 	Topic*
 1	   Version Control - GIT
 2	   Build Tool - Maven
 3	   Classes, Objects, OOP, Inner Classes/Nested Calsses, Lambda, Static Variable/Methods/Blocks/Data Types
 4	   Exceptions and Errors
 5	   Clean code
 6	   IO/Serialization
 7	   Collections
 8	   SOLID/KISS/DRY
 9	   Design Patterns
10	   Introduction TDD & Junit
11	   Logging & Log4J
12	   REST Architecture
13	   Multi Threading
14	   HTML & CSS
15	   Introduction JavaScript
16	   Introduction to Manual & Automation Testing & Test Pyramid
17	   Introduction to CI/CD
18	   DB Basics
  
"""
print('############ Reading Dataset ############')
# Import the dataset
dataset = pd.read_csv('./QA.csv', header = None, encoding = "utf-8")

# Split dataset into questions and answers
questions = dataset.iloc[:, 0].values
answers = dataset.iloc[:, 1].values

# Open Unanswered questions CSV for logging
#`UQ = open('Unanswered.csv', 'w')
print('############ trying to load model ############')
# Load Model
model = tf.saved_model.load(
    'use/',
    tags=None
)
print('############ model is now loaded ############')

# Get Embedding Matrix
embedding_matrix = ut.get_embeddings(model, questions)

# Setup Slack Client API
rtm_client = RTMClient(
  token = SLACK_TOKEN
  ,connect_method='rtm.start'
)

print('############ Starting RTM Client ############')


#rtm_client.start()

print('## Started ###')
# Start the client if it didn't start implicitly.
try:
    print('############ Inside Try ############')
    rtm_client.start()
    print('############ Exiting Try with No Exceptions ############')
except:
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  Bot is listening the chats! @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

# print('############ done with bot ! Should be listining now ############')

# Listen to MESSAGE Event
@RTMClient.run_on(event="message")
def reply_to_message(**payload):
    data = payload['data']
    
    if 'user' not in data or ut.is_bot(data['user'], BOT_ID):
        return
    
    webclient = payload['web_client']
    user = data['user']
    channel = data['channel']
    message = data['text']
    
    if ut.is_public(channel) and not ut.is_bot_tagged(message, BOT_ID_REGEX):
        return
    with open(str(pd.datetime.now().date())+'.txt','a') as logFile:
        logFile.write(str(pd.datetime.now())+' \t user : '+str(user)+'  Text: '+str(message)+'\n')
    
    message = ut.get_clean_message(message, BOT_ID_REGEX)
    
#    if ut.is_single_word(message):
#        if message in single_word_set:
#            reply = ' '.join(single_word_set[message])
#            ut.send_reply(user, webclient, channel, reply)
#            returnx
    if len(message)<2:
        ut.send_reply(user, webclient, channel," Hi ! Please ask Me Detailed Questions ")
        return
    
    message_embedding = ut.get_embeddings(model, [message])
    
    top_index, top_scores = ut.get_top_replies(embedding_matrix, message_embedding)
    top_replies = [answers[ind] for ind in top_index]
    top_q=[questions[ind] for ind in top_index]
    
    if top_replies[0] == '-1':
        ut.send_reply(user, webclient, channel,SYLLABUS)
        return
    
    if top_scores[0] < 0.5:
        with open('Unanswered.csv','a') as f:
            f.write(str(message)+',\n')
        ut.send_reply(user, webclient, channel, 'Sorry, I didn\'t get that!\nPlease elaborate your question')
        
    elif top_scores[0] > 0.5 and top_scores[0] < 0.6:
        with open('Unanswered.csv','a') as f:
            f.write(str(message)+',\n')
        if(len(top_q[0])<9):
            ut.send_reply(user, webclient, channel,top_replies[0])
        else:
            reply='\n*I Found These Matching Queries, Please see if it answers your question, else try elaborating your Question.*  \n'
            for index in range(len(top_q)):
                reply=reply+'\n'+ str(index+1)+'.  '+top_q[index]+' \n  '+top_replies[index]+' \n'
            reply=reply+'\n'+'*If your Query is still unanswered  please reach out to your college faculty*'
            ut.send_reply(user, webclient, channel, reply)
    
    else:
        ut.send_reply(user, webclient, channel, top_replies[0])

