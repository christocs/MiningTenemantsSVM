import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from gensim.models.word2vec import Word2Vec
from csv import reader

TRAIN_FILE = "SampleSetConditions.csv"
TEST_FILE = "TestSet.csv"

print("loading samples...",end="")
X, y = [], []
with open(TRAIN_FILE,"r",encoding="utf-8") as infile:
    csv_reader = reader(infile)
    next(csv_reader,None) #ignore header
    for row in csv_reader:
        X.append(row[4].split())
        y.append(row[1])
print("...done")

print ("total examples %s" % len(y))

print("vectorising...",end="")
model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.vectors)}
print("...done")


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(w2v))])
        else:
            self.dim=0
            
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


svm_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("SVM", SVC())])

print("fitting...",end="")
svm_w2v.fit(X,y)
print("...done")


#======TESTING======

unsorted_scores = (cross_val_score(svm_w2v, X, y, cv=5).mean())
print("cross validation check accuracy:",round(unsorted_scores*100,2),"%\n")

#read from testSet.csv
tests, correct_cat = [], []
with open(TEST_FILE,"r",encoding="utf-8") as infile:
    csv_reader = reader(infile)
    for row in csv_reader:
        tests.append(row[4].split())
        correct_cat.append(row[1])

predictions = svm_w2v.predict(tests)

correct = 0
for x in range(len(predictions)):
    if predictions[x] == correct_cat[x]:
        print("\t-Test",x,"correctly categorised as",predictions[x])
        correct+=1
    else:
        print("\t-Test",x,"FAILED. prediction:",predictions[x],"should be",correct_cat[x])

print("correct percentage:",round((correct/len(predictions))*100,2),"%")


