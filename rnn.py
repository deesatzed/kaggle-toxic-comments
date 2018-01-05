
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, BatchNormalization, GRU, Bidirectional, LSTM, GlobalMaxPool1D,Flatten
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
#io
train_df = pd.read_csv('/data/train.csv')
x_train = train_df.comment_text.fillna("_na_").values

output_file = "submission.csv"
classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(list(x_train))
#Get embed matrix from embedfile
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embed_size = 100 # how big is each word vector
EMBEDDING_FILE = '/data/glove.6B.100d.txt'
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
word_index = tokenizer.word_index
nb_words = min(20000, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= 20000: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
print("Done loading rnn")
class deepModel():
	def __init__(self):
		self.model = Sequential()
		self.model.add(Embedding(20000,embed_size,weights=[embedding_matrix]))
		self.model.add(Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
		self.model.add(GlobalMaxPool1D())
		self.model.add(BatchNormalization())
		self.model.add(Dense(50, activation='relu'))
		self.model.add(Dropout(0.1))
		self.model.add(Dense(6, activation='sigmoid'))
		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
	def run(self, X,y,X_te,full_X_te):
		print("Running LSTM...")
		# X = np.array(X)
		# y = np.array(y)
		# X_te = np.array(X_te)
		x_train = pad_sequences(tokenizer.texts_to_sequences(X),maxlen=100)
		x_test = pad_sequences(tokenizer.texts_to_sequences(X_te),maxlen=100)
		full_x_test = pad_sequences(tokenizer.texts_to_sequences(full_X_te), maxlen=100)
		y = y[classes].values
		print("Formatted inputs. Fitting...")
		file_path="weights_base.best.hdf5"
		checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
		early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
		callbacks_list = [checkpoint, early]
		#add validation split to test
		self.model.fit(x_train,y, batch_size=32, epochs = 2, callbacks = callbacks_list,validation_split = 0.2,verbose=2)
		self.model.load_weights(file_path)
		y_pred = self.model.predict([x_test],batch_size=1024, verbose=1)
		y_test = self.model.predict([full_x_test], batch_size=1024, verbose=1)
		# sample_submission = pd.read_csv("sample_submission.csv").head(len(y_test))
		# sample_submission[classes] = y_test
		print("LSTM done")
		return y_pred,y_test
class Finisher():
	def __init__(self,shape):
		self.model = Sequential()
		self.model.add(Conv1D(filters=32, kernel_size = 2, padding='same', activation='relu',input_shape = shape))
		self.model.add(BatchNormalization())
		self.model.add(Conv1D(filters=32, kernel_size = 2, padding='same', activation='relu'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling1D(pool_size=2))

		self.model.add(Dropout(0.2))

		self.model.add(Conv1D(filters=32, kernel_size = 2, padding='same', activation='relu'))
		self.model.add(BatchNormalization())
		self.model.add(Conv1D(filters=32, kernel_size = 2, padding='same', activation='relu'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling1D(pool_size=2))

		self.model.add(Dropout(0.2))
		self.model.add(Flatten())
		self.model.add(Dense(50, activation='relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(6, activation='sigmoid'))
		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
	def finish(self,X,y,t):
		file_path="weights_base.second.hdf5"
		checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
		early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
		callbacks_list = [checkpoint, early]
		#add validation split to test
		self.model.fit(X,y, batch_size=32, epochs = 30, callbacks = callbacks_list,validation_split = 0.2, verbose=2)
		self.model.load_weights(file_path)
		y_pred = self.model.predict([t],batch_size=1024, verbose=1)
		c = pd.read_csv("submission.csv")
		c[classes] = y_pred
		return c
