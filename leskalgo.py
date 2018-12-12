
# coding: utf-8

# In[710]:


from nltk.corpus import wordnet as wn
from difflib import SequenceMatcher 
from collections import Counter
from sklearn.model_selection import ParameterGrid
import math
import numpy as np
from sklearn import preprocessing
import operator
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


# In[711]:


'''
Created on Oct 26, 2015

@author: jcheung
'''
import xml.etree.cElementTree as ET
import codecs

class WSDInstance:
    def __init__(self, my_id, lemma, context, index):
        self.id = my_id         # id of the WSD instance
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.index = index      # index of lemma within the context
    def __str__(self):
        '''
        For printing purposes.
        '''
        return '%s\t%s\t%s\t%d' % (self.id, self.lemma, ' '.join(self.context), self.index)

def load_instances(f):
    '''
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    '''
    tree = ET.parse(f)
    root = tree.getroot()
    
    dev_instances = {}
    test_instances = {}
    
    for text in root:
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances
        for sentence in text:
            # construct sentence context
            context = [to_ascii(el.attrib['lemma']) for el in sentence]
            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    lemma = to_ascii(el.attrib['lemma'])
                    instances[my_id] = WSDInstance(my_id, lemma, context, i)
    return dev_instances, test_instances

def load_key(f):
    '''
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys. 
    '''
    dev_key = {}
    test_key = {}
    for line in open(f):
        if len(line) <= 1: continue
        #print line
        doc, my_id, sense_key = line.strip().split(' ', 2)
        if doc == 'd001':
            dev_key[my_id] = sense_key.split()
        else:
            test_key[my_id] = sense_key.split()
    return dev_key, test_key

def to_ascii(s):
    # remove all non-ascii characters
    return codecs.encode(s, 'ascii', 'ignore')


# In[712]:


regex_tokenizer = RegexpTokenizer(r'[\w_-]+')
wordnet_lemmatizer = WordNetLemmatizer()

def create_vector(text):
    text = ' '.join(text)
    tokens = regex_tokenizer.tokenize(text)
    tokens = [x.lower() for x in tokens]
    tokens = [wordnet_lemmatizer.lemmatize(x) for x in tokens]
    tokens = [x for x in tokens if x not in punctuation]
    tokens = [wordnet_lemmatizer.lemmatize(x) for x in tokens]
    return tokens
    
## thank you to https://stackoverflow.com/questions/15173225/how-to-calculate-cosine-similarity-given-2-sentence-strings-python
def get_cosine(string1, string2):
    
    vec1 = Counter(create_vector(string1))
    vec2 = Counter(create_vector(string2))
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator
    
def DistJaccard(string1, string2):
    str1 = set(string1)
    str2 = set(string2)
    return float(len(str1 & str2)) / len(str1 | str2)

def get_longest_common_substring(string1, string2):
    string1 = ' '.join(string1)
    match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
    return match.size   


# In[713]:


def addDefs(synset, relation, definition):
    relatives = []
    if 'hypernym' in relation:
        relatives = synset.hypernyms()
    
    if 'homonym' in relation:
        relatives = synset.homonyms()
        
    if relatives is None:
        return definition
    else:
        for rel in relatives:
            definition = definition + " " + rel.definition()
        
    return definition


# In[714]:


def compute_similarity(similarity_measure, context, definition):
    
    if "longest_common_substring" in similarity_measure:
        return get_longest_common_substring(context, definition)
    
    if "nltk_basic" in similarity_measure:
        return set(vectorize(definition)).intersection(set(context))
    
    if "cosine_similarity" in similarity_measure:
        return get_cosine(context, definition)
    
    if "jaccard_index" in similarity_measure:
        return DistJaccard(context, definition)


# In[715]:


#thanks to https://stackoverf:low.com/questions/15551195/how-to-get-the-wordnet-sense-frequency-of-a-synset-in-nltk 
def get_frequency_dist(word):
    synsets = wn.synsets(word)

    sense2freq = {}
    for s in synsets:
      freq = 0  
      for lemma in s.lemmas():
        freq+=lemma.count()
      sense2freq[s] = freq

    return sense2freq


# In[716]:


def myLesk(context, lemma, index, threshold, concatenate_hypernym_in_def, concatenate_hyponym_in_def, similarity_measure):
      
    synsets = wn.synsets(lemma)

    #if nothing interesting is happening, deal with that
    if not synsets:
        return None
    if len(synsets) == 1:
        return synsets[0]
    
    max_value = 0
    lesk_best_synset = None
    
    total_synset_count = sum([sum([l.count() for l in s.lemmas()]) for s in synsets])

    if total_synset_count == 0:
            return synsets[0]
        
    context = list(context)
    context = context[: index] + context[index+1 :]
    context = [word for word in context if word not in stopwords.words('english')]
    context = [x for x in context if x not in punctuation]
    context = [word.lower() for word in context]
        
    for synset in synsets:
        
        definition = synset.definition()
        
        if concatenate_hypernym_in_def:
            definition = addDefs(synset, 'hypernym', definition)
            
        if concatenate_hyponym_in_def:
            definition = addDefs(synset, 'hyponym', definition)
                    
        similarity = compute_similarity(similarity_measure, context, definition)
        
        if similarity > max_value:
            max_value = similarity
            lesk_best_synset = synset 
    
    if lesk_best_synset is None:
        return synsets[0]
    
    synset_count = float(sum([l.count() for l in lesk_best_synset.lemmas()]))

    #if what I've predicted is common enough, return it, otherwise return the most frequent synset
    if synset_count/total_synset_count > threshold:
        return lesk_best_synset
    else:
        return synsets[0]
    
    #lemma_frequency_dist = get_frequency_dist(lemma)
   ## print lemma_frequency_dist
   # try:
   #     normalized_lemma_frequency_dist_values = [float(i)/sum(lemma_frequency_dist.values()) for i in lemma_frequency_dist.values()]
   #  #   print normalized_lemma_frequency_dist_values
   #     variance =  np.var(normalized_lemma_frequency_dist_values)
   # except:
   #     variance = 0
    
   # return lesk_best_synset
      
   # if variance < variance_threshold:
   #     return lesk_best_synset
   # else:
   #     return most_frequent_synset(lemma)
    


# In[717]:


from nltk.wsd import lesk

data_f = 'multilingual-all-words.en.xml'
key_f = 'wordnet.en.key'
dev_instances, test_instances = load_instances(data_f)
dev_key, test_key = load_key(key_f)
    
    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
dev_instances = {k:v for (k,v) in dev_instances.iteritems() if k in dev_key}
test_instances = {k:v for (k,v) in test_instances.iteritems() if k in test_key}

#Converting sense keys to synsets
for item in dev_key:
    dev_key[item] = [wn.lemma_from_key(x).synset() for x in dev_key[item]]
for item in test_key:
    test_key[item] = [wn.lemma_from_key(x).synset() for x in test_key[item]]                 


# In[718]:


#parameter_optimization
param_grid = {'pos_var': [0.0, 0.2, 0.4, 0.6, 0.8], 'hyper' : [True, False], 'hypo' : [True, False],'sim' : ["longest_common_substring", "nltk_basic", "cosine_similarity", "jaccard_index"] }

grid = ParameterGrid(param_grid)

for params in grid:
    
    correct = 0.0
    total = 0.0
          
    for item in dev_instances:
        total += 1
        lemma = dev_instances[item].lemma
        context = dev_instances[item].context
        item_id = dev_instances[item].id
        index = dev_instances[item].index
        
        result = myLesk(context, lemma, index, params['pos_var'], params['hyper'], params['hypo'], params['sim'])

        #print result
        
        if result is None:
            continue
            
        if result in dev_key[item_id]:
            correct += 1
    
    print correct/total, params
 
                    
                    


# In[723]:


methods = ['baseline', 'nltk', 'myLesk'] #make this into objects in the future

#number_correct = 0.0
total = 0.0

baseline_correct = 0.0
nltk_correct = 0.0
mylesk_correct = 0.0
    
#FINAL EVALUATION
#dev_instances = test_instances
#dev_key = test_key
for item in dev_instances:
    
    for x in methods:
    
        lemma = dev_instances[item].lemma
        context = dev_instances[item].context
        item_id = dev_instances[item].id
    
        if x == 'baseline':
            lemma_frequency_dist = get_frequency_dist(lemma)
            try:
                normalized_lemma_frequency_dist_values = [float(i)/sum(lemma_frequency_dist.values()) for i in lemma_frequency_dist.values()]
                variance = np.var(lemma_frequency_dist.values())

            except:
                variance = 0
                
            result_baseline = wn.synsets(lemma)[0]
            
        if x == 'nltk':
            result_nltk = lesk(context, lemma)
            
        if x == 'myLesk':
            result_lesk = myLesk(context, lemma, index, 0.8, False, False, 'ntlk_basic')

            
    
    if result_nltk in dev_key[item_id]:
        nltk_correct +=1

    if result_baseline in dev_key[item_id]:
        baseline_correct +=1
        
    if result_lesk in dev_key[item_id]:
        mylesk_correct += 1

    total += 1

#    if result_nltk in dev_key[item_id] and result_baseline not in dev_key[item_id]:
#        #print "1"
#        #print variance
#        nltk_correct +=1
#        if variance is not 0:
#            var_list_1.append(variance)
#            
#        
#    elif result_nltk in dev_key[item_id] and result_baseline in dev_key[item_id]:
#        #print "2"
#        if variance is not 0:
#            var_list_2.append(variance)
#      
#    elif result_nltk not in dev_key[item_id] and result_baseline in dev_key[item_id]:
#        #print "3"
#        if variance is not 0:
#            var_list_3.append(variance)
#            
#        print result_nltk, result_baseline, dev_key[item_id]
#
#        
#    elif result_nltk not in dev_key[item_id] and result_baseline not in dev_key[item_id]:
#        #print "4"
#        if variance is not 0:
#            var_list_4.append(variance)

        
#for l in [var_list_1, var_list_2, var_list_3, var_list_4]:
#    print len(l)
#    print sum(l)/len(l)


   # else:
        #print dev_key[item_id], "NOOO", lemma_frequency_dist
    
print total
    
print baseline_correct/total
print nltk_correct/total
print mylesk_correct/total

