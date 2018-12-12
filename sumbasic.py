# encoding=utf8

#author: Caitrin Armstrong 260501112

import nltk
import sys
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

class Sentence:
    def __init__(self, sentence_text):
        self.sentence_text = sentence_text
        self.tokens = []
        self.score = 0.0

def read_articlecluster(files):
    sentences = []
    for f in files:
        with open(f, 'r') as infile:
            all_lines = infile.read()

        all_lines = all_lines.decode('utf-8')
        raw_sentences = sent_tokenize(all_lines)
        for raw_sentence in raw_sentences:
            sentences.append(Sentence(raw_sentence))
    return sentences

def preprocess(sentence):
       
    sentence_text = sentence.sentence_text
    #tokenize, with a tokenizer that only accepts text characters
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence_text.strip().lower())

    # Remove stop words
    stpwrds = stopwords.words('english')
    tokens = [token for token in tokens if token not in stpwrds]
    
    #lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    sentence.tokens = tokens

# deal with all sentences in the cluster
def preprocess_articlecluster(sentence_list):
    for sentence in sentence_list:
        preprocess(sentence)
    return sentence_list

#given all words in the cluster, calculate the prob of each word
def create_unigram_model(preprocessed_sentence_list):
    unigram_dict = {}
    all_tokens = []
    for sentence in preprocessed_sentence_list:
        all_tokens.extend(sentence.tokens)
    token_counts = Counter(all_tokens)
    for token in set(all_tokens):
        unigram_dict[token] = float(token_counts[token])/len(all_tokens)
        
    return unigram_dict

def redundancy_update(unigram_dict, tokens):
    for token in tokens:
        unigram_dict[token] = unigram_dict[token]**2
    return unigram_dict

#choose the best sentence according to its score
def rank_sentences(preprocessed_sentence_list, unigram_dict):
    for sentence in preprocessed_sentence_list:
        if len(sentence.tokens) < 1:
            continue
        score = 0.0
        for token in sentence.tokens:
            score += unigram_dict[token]
        sentence.score = score/len(sentence.tokens)

    ranked_list = sorted(preprocessed_sentence_list, key=lambda x: x.score, reverse=True)
    return ranked_list

def leading(filelist, limit):

    #look for the file with the shortest length
    min_length = 10000000
    min_file = ''
    for f in filelist:
        with open(f, 'r') as infile:
            text = infile.read()
            text_len = len(text)
        if text_len < min_length:
            min_length = text_len
            min_file = f

    with open(min_file, 'r') as infile:
        paragraphs = infile.readlines()
        paragraphs = [p for p in paragraphs if p != '\n']
          
        length = 0
        summary = []
        while True:
            paragraph = paragraphs.pop(0)
            paragraph_decoded = paragraph.decode('utf-8')

            best_sentence = sent_tokenize(paragraph_decoded)[0]
            length += len(best_sentence.split())
            if length > limit:
                break
            summary.append(best_sentence)
        return summary
    
    
def original(filelist, limit):
    sentences = read_articlecluster(filelist)
    preprocessed_sentence_list = preprocess_articlecluster(sentences)
    unigram_model = create_unigram_model(preprocessed_sentence_list)
    ranked_sentence_list = rank_sentences(preprocessed_sentence_list, unigram_model)
    
    length = 0
    summary = []

    #keep going until you run out of space
    while True:
        best_sentence = ranked_sentence_list.pop(0)
        length += len(best_sentence.tokens)
        if length > limit:
            break
        summary.append(best_sentence.sentence_text)
        unigram_model = redundancy_update(unigram_model, best_sentence.tokens)
        ranked_sentence_list = rank_sentences(preprocessed_sentence_list, unigram_model)
        
    return summary

def simplified(filelist, limit):
    
    sentences = read_articlecluster(filelist)
    preprocessed_sentence_list = preprocess_articlecluster(sentences)
    unigram_model = create_unigram_model(preprocessed_sentence_list)
    ranked_sentence_list = rank_sentences(preprocessed_sentence_list, unigram_model)
    
    length = 0
    summary = []

    #keep going until you run out of space
    while True:
        best_sentence = ranked_sentence_list.pop(0)
        length += len(best_sentence.tokens)
        if length > limit:
            break
        else:
            summary.append(best_sentence.sentence_text)
    return summary

def main():
    methodname = sys.argv[1]
    filelist = sys.argv[2:]

    if "orig" in methodname:
        summary = original(filelist, 100)
        print ' '.join(summary).encode('utf-8')

    elif "simplified" in methodname:
        summary = simplified(filelist, 100)
        print ' '.join(summary).encode('utf-8')

    elif "leading" in methodname:
        summary = leading(filelist, 100)
        print ' '.join(summary).encode('utf-8')

    else:
        print "no summary returned. check your method name and file paths"

if __name__ == "__main__":
    main()