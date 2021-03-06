import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from string import punctuation
import re
import lightgbm
import random

# remove random amount of characters from beginning or end of excerpt
# ------------------------------------
def remove_char(text):
    num = random.randint(15,35)
    num2 = random.randint(0,1)
    
    if num2 == 0:
        return text+text[0:num]
    else:
        return text[:-num]

# create new target value based off of standard error and selected target value
# ------------------------------------
def target_jitter(df):
    lst = list(zip(df.target, df.standard_error))
    final_lst = []
    for val in lst:
        num = random.randint(0,1)
        error = random.uniform(0, val[1])
        if num ==0:
            final_lst.append(val[0]+error)
        else:
            final_lst.append(val[0]-error)
    df['target'] = final_lst
    return df
    
# returns dictionary of words length counts
# ------------------------------------
def word_length_counts(text):
    '''
        Return: 
            Dictionry of words length counts 
    '''
    
    d = {"1_letter": 0, 
         "2_letter": 0, 
         "3_letter": 0, 
         "4_letter": 0, 
         "5_letter": 0, 
         "6_letter": 0, 
         "7_letter": 0, 
         "8_letter": 0, 
         "9_letter": 0, 
         "10_letter": 0, 
         "11_letter": 0, 
         "12_letter": 0
    }

    text.replace('\n', '')
    text = text.lower()
    n_text = ''
    for c in text: 
        if c not in punctuation:
            n_text += c
        else: n_text += ' '

    text = n_text
    for word in text.split(' '): 
        wl = len(word)
        if wl > 0:
            if len(word) >= 12: key = f"12_letter"
            else:  key = f"{len(word)}_letter"

            d[key] += 1; 
    return d

# returns Dictionary of source wiki, article, book, details, story or stories, kid, edu, simple
# ------------------------------------
def source_info(text):
    '''
        Return: 
            Dictionary of source wiki, article, book, details, story or stories, kid, edu, simple
    '''
    d = dict()
    source_type = ['wiki', 'article', 'book', 'details', 'kid', 'edu', 'simple', 'story', 'stories']
    
    for t in source_type:
        if t != 'stories': 
            d[t] = 0
            
        if t in text:
            if t == 'stories': 
                d['story'] = 1;
            else: d[t] = 1
    return d

# Return Dictionary of document length, word count, sentence count, average word lenght
# ------------------------------------
def document_info(text):
    '''
        Return:
            Dictionary of document lenght, word count, sentence count, average word lenght
    '''
    text_lenght = len(text)
    text = text.replace('\n', ' ')
    text_word_count = len(text.split(' '))
    text_sentence_count = len(re.split('\.|!|\?',text)) # modified by James
    text = re.split('\.|!|\?',text)
    text_avg_word_length = round(sum([len(t) for t in text]) / text_word_count, 2)

    document_info = {
        'doc_len': text_lenght,
        'word_count': text_word_count,
        'sent_count': text_sentence_count,
        'avg_word_len': text_avg_word_length
    }

    return document_info

# Returns the number of words per sentence in an excerpt
# ----------------------------------------
def words_per_sentence(text):
    total = []
    text = text.replace('Mrs.',"Mrs")
    text = text.replace('Mr.',"Mr")
    text = text.replace('Dr.',"Dr")
    text = text.replace('Capt.',"Capt")

    sentences = re.split('\.|!|\?',text)
    for sentence in sentences: #iterate over list of sentences
        if sentence != '':
            word_list = sentence.split(' ') #split a sentence into list of words
            while("" in word_list):
                word_list.remove("")
            while('"' in word_list):
                word_list.remove('"')
            total.append(len(word_list)) #total number of words in a sentence add to list
    return np.mean(total)

# Returns dictionary of counts of all characters in text
# ------------------------------------
def character_counts(text):
    '''
        Return:
            Dictionary of counts of all characters in text
    '''
    char_dict = dict()
    text = text.lower()
    text = text.replace('\n', ' ')
    
    for char in text:
        if char not in char_dict:
            char_dict[char] = 0
        
        char_dict[char] += 1     
        
    return char_dict

#Returns dictionary of all phonemic in text
# ------------------------------------
def phonemes_counts(text):
    '''
        Return: 
            Dictionay of all phonemic in text
    '''
    phonemes = ['ck', 'cc', 'di', 'nn', 'dd', 'ai', 'ss', 'mn', 'bb', 
                'sci', 'ze', 'qu', 'se', 'sc', 'ci', 'ps', 'si', 'tch', 
                'ngue', 'st', 'gu', 'th', 'pn', 've', 'te', 'zz', 'au', 
                'lm', 'lf', 'ge', 'wh', 'tu', 'wr', 'ph', 'sh', 'mm', 'gh', 
                'dge', 'ft', 'tt', 'ed', 'ng', 'lk', 'ti', 'gue', 'rr', 'ch', 
                'll', 'gn', 'ff', 'gg', 'pp', 'rh', 'ce', 'mb', 'kn', 
                'eer', 'ere', 'uy', 'ho', 'ear', 'ei', 'ar', 'ai', 
                'oor', 'ure', 'eigh', 'ey', 'is', 'ae', 'ow', 'or', 'ew', 
                'ore', 'ur', 'uoy', 'air', 'au', 'ough', 'yr', 
                'ea', 'ayer', 'augh', 'aw', 'eau', 'aigh', 'igh', 'oy', 
                'oo', 'ue', 'are', 'ee', 'oa', 'et', 'y', 'er', 'eir', 
                'oew', 'oar', 'ie', 'eo', 'ui', 'ier', 'ou', 'ir', 'oi', 
                'ay', 'ye', 'oe', 'our']
    temp_dict = dict()
    
    # lower text 
    text = text.lower()
    
    for p in phonemes:
        temp_dict[p] = text.count(p)
    
    return temp_dict

# Returns new training set with additional data based off on excerpts with target value less than float1 and greater than float2
#-----------------------
def add_data(df,float1,float2):
    group = df.loc[(df.target<float1) | (df.target>float2)]
    group = group.copy()
    group['mod'] =  group.excerpt.apply(lambda x: remove_char(x))
    group.drop(columns =['excerpt'],axis = 1,inplace = True)
    group = group.rename(columns = {"mod":"excerpt"})
    target_jitter(group)
    mod_train = pd.concat([df,group],sort = 'False')
    train = mod_train
    return train


# Processing Pipeline Function
# -------------------------------------------------------
# -------------------------------------------------------
def processing_pipeline(train, test):

    train = add_data(train,-2,0)
    train = add_data(train,-5,1)
    train = add_data(train,-5,2)
    train = add_data(train,-5,.5)
    train = add_data(train,-5,.5)
    train = add_data(train,-5,.5)
    train = add_data(train,-2,6)
    train = add_data(train,-2,6)

    print(train.target.plot.hist())
    print(train.target.mean())
    print(len(train))

    #combine train and test set for pre-processing
    c_df = pd.concat([train, test], axis = 0)

    #apply functions to train and test set
    series_documents =  c_df.excerpt.apply(lambda x: document_info(x))
    series_phonemes = c_df.excerpt.apply(lambda x: phonemes_counts(x))
    series_characters =  c_df.excerpt.apply(lambda x: character_counts(x))
    series_word_length = c_df.excerpt.apply(lambda x: word_length_counts(x))
    series_word_per_sent = c_df.excerpt.apply(lambda x: words_per_sentence(x))

    df_documents = pd.DataFrame(list(series_documents))
    df_phonemes = pd.DataFrame(list(series_phonemes))
    df_characters = pd.DataFrame(list(series_characters))
    df_word_length = pd.DataFrame(list(series_word_length))
    df_word_per_sent = pd.DataFrame(list(series_word_per_sent))
    df_word_per_sent.rename(columns={0:'wps'}, inplace = True)

    #create column that returns 1 if excerpt has dialogue or 0 if no dialogue
    df_dialogue = pd.DataFrame(list(c_df.excerpt.str.contains("\".*\"",regex = True).astype(int)))
    df_dialogue.rename(columns = {0:'dialogue'},inplace = True)

    #turn na's to 0
    df_characters.fillna(0, inplace = True)

    #turn data to integer type
    for col in df_characters.columns:
        df_chracters[col] = df_chracters[col].astype(int, copy=False)

    #create processed dataset for model
    df_X = pd.concat([df_documents, df_phonemes, df_word_length,df_word_per_sent,df_dialogue], axis = 1)

    #split dataset into train and test set
    df_X_test = df_X.iloc[-len(test):]
    df_X = df_X.iloc[:-len(test)]
    df_y = c_df['target'][:-len(test)] 

    return df_X, df_X_test, df_y


# #Standardize Data
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import cross_validate

# scaler = MinMaxScaler()
# print(scaler.fit(df_X))
# x_train = scaler.transform(df_X)