#NOT my code. https://www.kaggle.com/thousandvoices/logistic-regression-with-words-and-char-n-grams/code
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('/data/train.csv').fillna(' ')
test = pd.read_csv('/data/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,1),
    max_features=20000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 4),
    max_features=20000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack((train_char_features, train_word_features))
test_features = hstack((test_char_features, test_word_features))
def run(i):
    losses = []
    predictions = {'id': test['id']}
    for class_name in class_names:
        train_target = train[class_name]
        classifier = LogisticRegression(C=4.0, solver='sag')

        cv_loss = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='neg_log_loss'))
        losses.append(cv_loss)
        print('CV loss for class {} is {}'.format(class_name, cv_loss))

        classifier.fit(train_features, train_target)
        predictions[class_name] = classifier.predict_proba(test_features)[:, 1]

    print('Total CV loss is {}'.format(np.mean(losses)))

    submission = pd.DataFrame.from_dict(predictions)
    submission.to_csv('/output/submission{0}.csv'.format(i), index=False)
i = 0
while i<5:
    run(i)
    i = i + 1
