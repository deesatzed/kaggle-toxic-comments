#credits to https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout-lb-0-048/notebook
from scipy.sparse import hstack
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#definitions
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
vect = TfidfVectorizer(analyzer='char', ngram_range=(1,4), max_features=50000, min_df=2)
# word_vectorizer = TfidfVectorizer(
#     sublinear_tf=True,
#     strip_accents='unicode',
#     analyzer='word',
#     token_pattern=r'\w{1,}',
#     ngram_range=(1,1),
#     max_features=20000)
# char_vectorizer = TfidfVectorizer(
#     sublinear_tf=True,
#     strip_accents='unicode',
#     analyzer='char',
#     ngram_range=(1, 4),
#     max_features=20000)
train_df = pd.read_csv('/data/train.csv').fillna("_na_")
test_df = pd.read_csv('/data/test.csv').fillna("_na_")
train_text = train_df['comment_text']
test_text = test_df['comment_text']
all_text = pd.concat([train_text, test_text])
vect.fit(all_text)
# word_vectorizer.fit(all_text)
# char_vectorizer.fit(all_text)
# vect.fit(x_train)
class LRModel():
	def __init__(self):
		self.model = LogisticRegression(C=4.0, solver='sag' )
	def run(self,X,y,X_te,full_X_te):
		print("running lr model")
# 		x_train_dtm_o = word_vectorizer.transform(X)
# 		x_test_dtm_o = word_vectorizer.transform(X_te)
# 		full_x_test_o = word_vectorizer.transform(full_X_te)
# 		x_train_dtm_i = char_vectorizer.transform(X)
# 		x_test_dtm_i = char_vectorizer.transform(X_te)
# 		full_x_test_i = char_vectorizer.transform(full_X_te)
# 		x_train_dtm = hstack((x_train_dtm_o, x_train_dtm_i))
# 		x_test_dtm = hstack((x_test_dtm_o, x_test_dtm_i))
# 		full_x_test = hstack((full_x_test_o, full_x_test_i))
		x_train_dtm = vect.transform(X)
		x_test_dtm = vect.transform(X_te)
		full_x_test = vect.transform(full_X_te)
		submission = pd.read_csv("sample_submission.csv").head(x_test_dtm.shape[0])
		sub2 = pd.read_csv("sample_submission.csv").head(full_x_test.shape[0])
		for class_label in list_classes:
			print('Training {}'.format(class_label))
			self.model.fit(x_train_dtm, y[class_label].astype("int"))
			y_test = self.model.predict_proba(x_test_dtm)[:,1]
			y_pred = self.model.predict_proba(full_x_test)[:,1]
			submission[class_label] = y_test
			sub2[class_label] = y_pred
		return submission[list_classes].values,sub2[list_classes].values
class Finisher():
	def __init__(self):
		self.model = LogisticRegression(C=0.9)
	def finish(self,length,X,y,t):
		print("Running the 2nd level model..")
		#X,mixed predictions on X_train, # of training samples by 6 * 2 (# classifiers)
		#y, actual predictions of X_train, is number of test samples by 6 
		#t, new test set, # of test samples by 6 * 2
		#Return value should be submission loaded.
		#pseudocode: c = load submission.csv
		#for i in range len 6 [0 incl 5]:
			#a = load empty matrix (X.shape[0], length)
			#b = load empty matrix (T.shape[0], length)
				#for j in range len clf [0 incl 1]:
					# a[:, j] = X[:(i+1)*(j+1) -1]
					# b[:, j] = t[:(i+1)*(j+1) -1]
			# fit model on (a, y[i])
			# c[classes[i]] = model predict (b)
		#return c
		c = pd.read_csv("submission.csv")
		for i in range(6):
		# 0 1 2 3 4 5 
		# 0 1
			a = np.zeros((X.shape[0], length))
			b = np.zeros((t.shape[0], length))
			for j in range(length):
				a[:, j] = X[:, 6*j + i]
				b[:,j] = t[: , 6*j + i]
			self.model.fit(a,y[:,i])
			y_hell = self.model.predict_proba(b)[:,1]
			print(c[list_classes[i]].shape)
			print(y_hell.shape)
			c[list_classes[i]] = y_hell
		return c


