##Caitrin Armstrong
##260501112

import argparse
import re
import nltk
from nltk.tag import hmm
from nltk.tag.hmm import HiddenMarkovModelTagger
from nltk.probability import ConditionalFreqDist, FreqDist, MLEProbDist, ConditionalProbDist, LaplaceProbDist
import random

def main():
    parser = argparse.ArgumentParser(description='Text decipher options')
    parser.add_argument('cipher_folder', help='cipher data folder')
    parser.add_argument('--laplace', '-laplace', action='store_true', default=False, help='Laplace Smoothing')
    parser.add_argument('--langmod', '-lm', action='store_true', default=False, help='Improved decoder')

    args = parser.parse_args()
    cipher_folder = args.cipher_folder
    laplace = args.laplace
    langmod = args.langmod
    number_of_supp_lines = 100 #the more lines the slower the code!

    train_data, test_data, train_plain = get_data(cipher_folder)
    preprocess_supp_data()
    supp_data = read_preprocessed_supp_data(number_of_supp_lines)
    for line in train_plain: #this is so later we have all the transitions in the same place
        supp_data.extend(list(line))
   
    if laplace:
        smoothing = LaplaceProbDist
    else:
        smoothing = MLEProbDist

    trainer = hmm.HiddenMarkovModelTrainer()
    decoder = trainer.train_supervised(train_data, smoothing)

    #decoder_supp = trainer_supp.train_unsupervised(supp_data, update_outputs=False, model=decoder)
    #because there's a bug in train_unsupervised (although I found out how to fix it!), I will have to do this manually....
    #code copied from the nltk train_supervised method
    #here, we are updating the transition data to include our supplemental data
    if langmod:
        states = decoder._states
        symbols = decoder._symbols
        outputs = decoder._outputs
        priors = decoder._priors
        starting = FreqDist() #declaring
        transitions = ConditionalFreqDist() #declaring, why we needed all the transitions in the same place
        for item in supp_data:
            for sequence in supp_data:
                lasts = None
                for state in sequence:
                    if lasts is None:
                        starting[state] += 1
                    else:
                        transitions[lasts][state] += 1
                    lasts = state

        if laplace:
            estimator = LaplaceProbDist
        else:
            estimator = lambda fdist, bins: MLEProbDist(fdist) #getting this straight from the source code
    
        N = len(states)
        pi = estimator(starting, N)
        A = ConditionalProbDist(transitions, estimator, N)
        #conditionalPD is actually already defined by our previously trained model as outputs.
        #we don't have new ones!
        decoder = HiddenMarkovModelTagger(symbols, states, A, outputs, pi)

    print(decoder.test(test_data))
    for sent in test_data:
            print "".join([y[1] for y in decoder.tag([x[0] for x in sent])])

def get_data(cipher_folder):
    all_data = dict()
    for f in ["test_cipher", "test_plain", "train_cipher", "train_plain"]:
        with open(cipher_folder+"/"+f+".txt") as openf:
            lines = openf.read().splitlines()
            all_data[f] = [list(s) for s in lines]

    train_data = list()
    test_data = list()
    for i in range(len(all_data["train_cipher"])):
        train_data.append(zip(all_data["train_cipher"][i], all_data["train_plain"][i]))
    for i in range(len(all_data["test_cipher"])):
        test_data.append(zip(all_data["test_cipher"][i], all_data["test_plain"][i]))

    return train_data, test_data, all_data["train_plain"]


def preprocess_supp_data():
    f2 = open('preprocessed_supp_data.txt', 'w')
    supp_data = list()
    files = ["rt-polaritydata/rt-polarity.neg.txt", "rt-polaritydata/rt-polarity.pos.txt"]
    for file in files:
        with open(file) as openf:
            lines = openf.read().splitlines()
            for line in lines:
                 line = re.sub('[^a-zA-Z,.\s]', '', line.strip().lower()) #TODO: save this to a file
                 f2.write(line+"\n")
    f2.close()

#hardcoded because I'm assuming you're not changing it (hi!)
def read_preprocessed_supp_data(number_of_lines):
    supp_data = list()
    with open('preprocessed_supp_data.txt', 'r') as openf:
        lines = openf.read().splitlines() #ok because small file
        for line in lines:
            supp_data.append(list(line)) #turns it into characters
    return random.sample(supp_data, number_of_lines)

if __name__ == "__main__":
    main()
