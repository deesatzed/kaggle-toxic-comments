from init import LRModel#,Finisher
from rnn import deepModel, Finisher
from xgbclassifier import XClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
x_train = train_df.comment_text.fillna("_na_").values
x_test = test_df.comment_text.fillna("_na_").values
classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
class Ensemble():
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
    def fit_predict(self, X, y, T):
        X = np.array(X)
        y_np = y.values
        T = np.array(T)
        folds = list(KFold(n_splits=self.n_folds, shuffle=True, random_state=2017).split(X))
        S_train = np.zeros((X.shape[0], 6*len(self.base_models)))
        S_test = []
        final_test = np.empty((T.shape[0],6*len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            for j, (train_idx, test_idx) in enumerate(folds):
                print(train_idx)
                print("On {}{}".format(i,j))
                X_train = X[train_idx]
                y_train = y_np[train_idx]
                X_holdout = X[test_idx]
                y_pred, y_test = clf.run(X_train,pd.DataFrame(y_train, index=y.index[train_idx], columns=y.columns),X_holdout,T)[:]
                S_train[test_idx, i*6:i*6+6] = y_pred
                S_test.append(y_test)
            final_test[:,i*6:i*6+6] = np.mean(np.array(S_test), axis=0)
        return self.stacker.finish(np.expand_dims(S_train, axis=2), y[classes].values, np.expand_dims(final_test, axis=2))[:]
go = Ensemble(5, Finisher((12,1)), [LRModel(), deepModel()])
go.fit_predict(x_train, train_df, x_test).to_csv('try.csv', index=False)