# dependencies and models used.
import sklearn 
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import FunctionTransformer
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error

# function for reading data in a .tsv file.
def read_data(filename):
    X = []
    Y = []
    with open(filename, encoding ="UTF8") as f:
        for l in f:
            cols = l.strip('\n.').split('\t')
            X.append(cols[1])
            Y.append(cols[0])
    return X, Y


''' 
# Create a new file: "preprocessed-brexit.tsv". The aim is to 
# filter out comments with large disagreement between annotators
# and store the preprocessed data in this file for later use.'''

prepros = open("preprocessed-brexit.tsv","w+", encoding ="UTF8")

# Read the raw data
XtrainStart, YtrainStart = read_data("train-data.tsv")

'''
# Commulatively processes the data and adds it to the tmpData variable.
# If the agreement between annotators is below 65%, the data is disregarded.'''

tmpData = ""
for x, y in zip(XtrainStart, YtrainStart):
    tmp = y.split('/')
    if((tmp.count('1') / len(tmp) > 0.65)):
        tmpData += "1\t" + x + "\n"
    elif((tmp.count('0') / len(tmp) > 0.65)):
        tmpData += "0\t" + x + "\n"

# writes to file.
prepros.write(tmpData)
prepros.close()

'''
# An extra touch added is that the comment length could be an interesting feature
# added to the documents.'''

def add_features(X):
    for i, x in enumerate(X):
        if(len(x) < 32):
            X[i] = x + " tlkxas39432r"
        else:
            X[i] = x + " kjslf233209"
    return X

# Read finalized training and test data.
Xtrain, Ytrain = read_data("preprosessed-brexit.tsv")
Xtest, Ytest = read_data("test-data.tsv")

# kbest features to classify the data. This makes use of the statistical effect size
# from an analysis of variance between the features.
kbest = SelectKBest(f_classif)

# make a few different pipelines to see which performs best.
naive_pipeline = Pipeline([('tfid', TfidfVectorizer()), ('std', StandardScaler(with_mean=False)), ('kbest', SelectKBest()), ('naive', BernoulliNB())])
dummy_pipeline = Pipeline([('tfid', TfidfVectorizer()), ('kbest', SelectKBest()), ('dummy', DummyClassifier())])
linSVC_pipeline = Pipeline([('tfid', TfidfVectorizer()), ('kbest', SelectKBest()), ('linSVC', LinearSVC())])
theNewest_pipeline = Pipeline([('tfid', TfidfVectorizer()), ('kbest', SelectKBest()), ('linSVC', LinearSVC())])

# grid search for the best set of "k-best" features to use.
# having too many values here will drastically increase the computation time.
naive_param_grid = {'kbest__k': [ 2000, 2500, 2400, 2600, 2300, 2700]}, {'naive__alpha': [1.0, 0.99]}
dummy_param_grid = {'kbest__k': [ 2000, 2500, 2400, 2600, 2300, 2700]}
linSVC_param_grid = [{'kbest__k': [ 2000,3000,4000,5000]}]

naive_grid_search = GridSearchCV(naive_pipeline, naive_param_grid)
dummy_grid_search = GridSearchCV(dummy_pipeline, dummy_param_grid)

# finally train data on our select seeming best model
naive_grid_search.fit(Xtrain, Ytrain)
naive_grid_search_pred = naive_grid_search.best_estimator_.predict(Xtrain)


dummy_grid_search.fit(Xtrain, Ytrain)
linSVC_grid_search = GridSearchCV(linSVC_pipeline, linSVC_param_grid)
linSVC_grid_search.fit(Xtrain, Ytrain)

print("\n\n\n")
print("score on test sample: ")
print(naive_grid_search.score(Xtest, Ytest))
print("\n\n\n")
print("confusion matrix: ")
print(confusion_matrix(Ytrain, naive_grid_search_pred))
print("\n\n\n")
print("cross_validation\n")
print(cross_validate(naive_grid_search, Xtrain, Ytrain, cv=6))
print("\n\n\n")
