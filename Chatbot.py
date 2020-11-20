#!/usr/bin/env python
# coding: utf-8

# In[21]:



import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from gtts import gTTS
import os
warnings.filterwarnings('ignore')
import speech_recognition as sr 
import nltk
from nltk.stem import WordNetLemmatizer
from playsound import playsound

# for downloading package files can be commented after First run
#nltk.download('popular', quiet=True)
#nltk.download('nps_chat',quiet=True)
#nltk.download('punkt') 
#nltk.download('wordnet')


posts = nltk.corpus.nps_chat.xml_posts()[:10000]

# To Recognise input type as QUES. 
def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Recognised input types 
#Greet
#"Bye"/>
#"Clarify"/>
#"Continuer"/>
#"Emotion"/>
#"Emphasis"/>
#"Greet"/>
#"Reject"/>
#"Statement"/>
#"System"/>
#"nAnswer"/>
#"whQuestion"/>
#"yAnswer"/>
#"ynQuestion"/>
#"Other"


#colour palet
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk)) 
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk)) 
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk)) 
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk)) 
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk)) 
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))


 
#Reading in the input_corpus
with open('data','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response and processing 
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


#Recording voice input using microphone 
file = "file1.mp3"
flag=True
fst="My name is Jarvis. I will answer your queries about Science. If you want to exit, say Bye"
tts = gTTS(fst, lang='en')
tts.get_urls()

tts.save(file)

prYellow(fst)

playsound(file)
os.remove(file)
r = sr.Recognizer()

# Taking voice input and processing 
while(flag==True):
#     with sr.Microphone() as source:
#         audio= r.listen(source)
#     try:
#         user_response = format(r.recognize(audio))
#         print("\033[91m {}\033[00m" .format("YOU SAID : "+user_response))
#     except sr.UnknownValueError:
#         prYellow("Oops! Didn't catch that")
#         pass
    
    user_response = input()
    user_response=user_response.lower()
    clas=classifier.classify(dialogue_act_features(user_response))
    if(clas!='Bye'):
        if(clas=='Emotion'):
            flag=False
            new="Jarvis: You are welcome.."
            n = gTTS(new, lang='en')
            
            file4="new.mp3"
            n.save(file4)
            prYellow(n)
            os.remove(n)
        else:
            if(greeting(user_response)!=None):
                greet="greet.mp3"
                grt=greeting(user_response)
                print("\033[93m {}\033[00m" .format("Jarvis: "+grt))
                grt1 = gTTS(grt, lang='en')
                grt1.save(greet)
                playsound(greet)
                os.remove(greet)
            else:
                print("\033[93m {}\033[00m" .format("Jarvis 1: ",end=""))
                res=(response(user_response))
                prYellow(res)
                res1 = gTTS(res, lang='en')
                sent_tokens.remove(user_response)
                file2="res.mp3"
                res1.save(file2)
                playsound(file2)
                os.remove(file2)
                
    else:
        flag=False
        b="Bye! take care.."
        bye = gTTS(b, lang='en')
        file3="bye.mp3"
        bye.save(file3)
        prYellow(b)
        playsound(file3)
        
        os.remove(file3)


# In[ ]:




