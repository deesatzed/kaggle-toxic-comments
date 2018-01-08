# thanks https://www.kaggle.com/tejasrinivas/xgb-baseline-0-056-on-lb-comments-classification/code
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

train = pd.read_csv('/data/train.csv')
test = pd.read_csv('/data/test.csv')

train = train.fillna("unknown")
test = test.fillna("unknown")


train_mes, valid_mes, train_l, valid_l = train_test_split(train['comment_text'],train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']], test_size=0.2, random_state=2)

'''
def text_process(comment):
    nopunc = [char for char in comment if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
'''
#Couldnt remove the stop words using the above function since it is taking too long
#Can try it on a local machine, I feel it improves the score-Not sure though


'''
transform_com = CountVectorizer().fit(pd.concat([train['comment_text'],test['comment_text']],axis=0))
comments_train = transform_com.transform(train['comment_text'])
comments_test = transform_com.transform(test['comment_text'])
gc.collect()'''

#Using the tokenize function from Jeremy's kernel
# import re, string
# re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
# def tokenize(s): return re_tok.sub(r' \1 ', s).split()

# transform_com = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
#                min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
#                smooth_idf=1, sublinear_tf=1).fit(pd.concat([train['comment_text'],test['comment_text']],axis=0))
# '''comments_train = transform_com.transform(train['comment_text'])'''
# comments_train = transform_com.transform(train_mes)
# comments_valid = transform_com.transform(valid_mes)
# comments_test = transform_com.transform(test['comment_text'])
# gc.collect()

word_vect = word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,1),
    max_features=20000)
char_vect = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 4),
    max_features=20000)
word_vect.fit(pd.concat([train['comment_text'],test['comment_text']],axis=0))
char_vect.fit(pd.concat([train['comment_text'],test['comment_text']],axis=0))
comments_train_word = word_vect.transform(train_mes)
comments_train_char = char_vect.transform(train_mes)
comments_valid_word = word_vect.transform(valid_mes)
comments_valid_char = char_vect.transform(valid_mes)
comments_test_word = word_vect.transform(test['comment_text'])
comments_test_char = char_vect.transform(test['comment_text'])
comments_train = hstack((comments_train_word,comments_train_char))
comments_valid = hstack((comments_valid_word,comments_valid_char))
comments_test = hstack((comments_test_word,comments_test_char))

import xgboost as xgb
'''
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=2, num_rounds=400):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.1
    param['max_depth'] = 5
    param['silent'] = 1
    param['eval_metric'] = 'logloss'
    param['min_child_weight'] = 1
    param['subsample'] = 0.4
    param['colsample_bytree'] = 0.6
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    xgtest = xgb.DMatrix(test_X)
    model = xgb.train(plst, xgtrain, num_rounds)
    pred_test_y = model.predict(xgtest)
    return pred_test_y, model
'''


def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=2017, num_rounds=400):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.12
    param['max_depth'] = 5
    param['silent'] = 1
    param['eval_metric'] = 'logloss'
    param['min_child_weight'] = 1
    param['subsample'] = 0.5
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return model
    

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
preds = np.zeros((test.shape[0], len(col)))

for i, j in enumerate(col):
    print('fit '+j)
    model = runXGB(comments_train, train_l[j], comments_valid,valid_l[j])
    preds[:,i] = model.predict(xgb.DMatrix(comments_test))
    gc.collect()

subm = pd.read_csv('/data/sample_submission.csv')    
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv('/output/xgb.csv', index=False)
    
