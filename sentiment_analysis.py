import numpy as np
from nltk.corpus import stopwords as sw
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
import pickle
import string
import pandas as pd
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import itertools

#import data
with open("rt-polaritydata/rt-polaritydata/rt-polarity.neg","r") as f_neg:
    neg = f_neg.readlines()
    neg = [s.translate(None, string.punctuation).strip() for s in neg] 
with open("rt-polaritydata/rt-polaritydata/rt-polarity.pos","r") as f_pos:
    pos = f_pos.readlines()
    pos = [s.translate(None, string.punctuation).strip() for s in pos]  

#make labels match data
neg_labels = ["n"] * len(neg)
pos_labels = ["p"] * len(pos)
documents = neg + pos
labels = neg_labels + pos_labels

#create test-train split: 80-20.
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, stratify=labels)

def get_feature_engineering_combos():
    
    # preprocessing options:
    # unigrams or (unigrams and bigrams)
    # remove stop words or do not remove stopwords
    # remove infrequently occuring words or do not remove infrequently occuring words
    # remove infrequently occuring bigrams or do not remove infrequently occuring bigrams (threshold: [1.0, 0.8, 0.6, 0.4, 0.2, 0.01])

    feature_engineering_combos = dict()
    # 0.01 because otherwise it conflicts with min_df
    for max_df_value in [1.0, 0.8, 0.6, 0.4, 0.2, 0.01]:
        feature_engineering_combos["cv_ug", max_df_value] = CountVectorizer(ngram_range=(1, 1), max_df=max_df_value, decode_error='ignore')
        feature_engineering_combos["cv_ug_bg ", max_df_value]= CountVectorizer(ngram_range=(1, 2), max_df=max_df_value, decode_error='ignore')
        feature_engineering_combos["cv_ug_stopwords_removed", max_df_value] = CountVectorizer(ngram_range=(1, 1), stop_words='english', max_df=max_df_value, decode_error='ignore')
        feature_engineering_combos["cv_ug_bg_stopwords_removed", max_df_value] = CountVectorizer(ngram_range=(1, 2), stop_words='english', max_df=max_df_value, decode_error='ignore')
        feature_engineering_combos["tf_ug", max_df_value] = TfidfVectorizer(ngram_range=(1, 1), max_df=max_df_value, decode_error='ignore')
        feature_engineering_combos["tf_ug_bg ", max_df_value]= TfidfVectorizer(ngram_range=(1, 2), max_df=max_df_value, decode_error='ignore')
        feature_engineering_combos["tf_ug_stopwords_removed", max_df_value] = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', max_df=max_df_value, decode_error='ignore')
        feature_engineering_combos["tf_ug_bg_stopwords_removed", max_df_value] = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_df=max_df_value, decode_error='ignore')

    return feature_engineering_combos

def grid_search(X_train, y_train):
    logistic = Pipeline((
        ('clf', linear_model.LogisticRegression()),
    ))
    
    svm = Pipeline((
        ('clf', LinearSVC()),
    ))
    
    multinomial = Pipeline((
        ('clf', MultinomialNB()),
    ))
    
    dummy = Pipeline((
        ('clf', DummyClassifier()),
    ))
    
    logistic_params = {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        #TODO: in the future also test for penalty?
    }
    
    svm_params  = {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    }
    
    multinomial_params = {
        'clf__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    }
    
    dummy_params = {
        
    }
    
    pipelines = [logistic, svm, multinomial, dummy]
    parameters = [logistic_params, svm_params, multinomial_params, dummy_params]
        
    results = []
    
    #for every one of the classifiers, 
    #run gridsearchcv which will search of the potential parameter space for the best-perfoming hyperparameters
    for i in range(len(pipelines)):
        
        gs = GridSearchCV(pipelines[i], parameters[i], n_jobs=1, cv=10)
        
        model = gs.fit(X_train, y_train)
           
        results.append(model)
    return results

#taken, with few modifications from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

"""
    now for each combination of feature transformation steps, previously created in feature_engineering_combos,
run gridsearch on each transformed X_train.
this is like creating a tree that branches out to all possible combinations of
transformations, their parameters, models and the model's parameters
"""
feature_engineering_combos = get_feature_engineering_combos()
feature_engineering_combos_persistent_keys = feature_engineering_combos.keys()
results = list()
transformers = list() #need to save this in order to transform X_test the same way later on
total = len(feature_engineering_combos_persistent_keys) * 3
count = 0
for combo in feature_engineering_combos_persistent_keys:
    print "starting " + str(count) + " of " + str(total)
    count += 1
    transformer = feature_engineering_combos[combo].fit(X_train)
    transformers.append(transformer)
    X_train_transformed = transformer.transform(X_train)
    results.append(grid_search(X_train_transformed, y_train))

#Now process results
classifiers = ["logistic", "svm", "multinomial", "dummy"]
best_score_so_far = 0
best_y_pred = None
for i in range(len(results)):
    for j in range(len(classifiers)):
        X_test_transformed = transformers[i].transform(X_test)
        test_score = results[i][j].best_estimator_.score(X_test_transformed, y_test)
        if test_score > best_score_so_far:
            best_score_so_far = test_score
            best_y_pred = results[i][j].best_estimator_.predict(X_test_transformed)
        print results[i][j].best_score_, "\t", test_score, "\t", classifiers[j], "\t", feature_engineering_combos_persistent_keys[i], "\t", results[i][j].best_estimator_.get_params()
        #res = pd.DataFrame(results[i][j].cv_results_)
        #display(res)
        
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, best_y_pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["negative", "positive"], normalize=True,
                      title='Confusion Matrix for the Best Predictor')
plt.savefig('confusion_matrix.png')
