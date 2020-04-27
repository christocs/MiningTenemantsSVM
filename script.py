import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import TreebankWordTokenizer 
  
TRAIN_SET_PATH = "Reuters-r8-no-stop.txt"

X, y = [], []
with open(TRAIN_SET_PATH, "r") as infile:
    for line in infile:
        label, text = line.split("\t")
        # texts are already tokenized, just split on space
        X.append(text.split())
        y.append(label)
X, y = np.array(X), np.array(y)
#X=list of lists text, y = associated category

print ("total examples %s" % len(y))

print("vectorising...")
model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.vectors)}
print("...done")


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec)>0:
            #self.dim=len(word2vec[next(iter(glove_small))])
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

print("fitting...")
svm_w2v.fit(X,y)
print("...done")

pre_test1 = "enex resources corp enex qtr loss shr not net loss profit revs year shr loss profit cts net loss profit revs reuter"
pre_test2 = "douglas computer international year end shr cts net revs reuter "
pre_test3 = "baker repeats hopes prime hike temporary treasury secretary james baker reiterated hope that this week rise prime rates temporary blip upwards hope that simply temporary blip upward past baker television interview cable news network interview airs tomorrow cnn released extracts remarks today baker repeated position that reaction financial markets tariffs japanese electronic goods showed importance united states not protectionist markets telling careful reuter "


tokenizer = TreebankWordTokenizer() 
test1=tokenizer.tokenize(pre_test1) 
test2=tokenizer.tokenize(pre_test2)
test3=tokenizer.tokenize(pre_test3)

tests = [test1, test2, test3]

print (svm_w2v.predict(tests))


'''
unsorted_scores = (cross_val_score(svm_w2v, X, y, cv=5).mean())
print(unsorted_scores)
'''
