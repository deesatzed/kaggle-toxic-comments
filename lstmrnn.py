#credits to several public kaggle kernels in the competition
import numpy as np, pandas as pd

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, BatchNormalization, GRU, Bidirectional, LSTM, GlobalMaxPool1D
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint

EMBEDDING_FILE = '/out/data/glove.840B.300d.txt'
train_file = "/data/train.csv"
test_file = "/data/test.csv"
classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
traintext = train['comment_text'].fillna("_na_")
testtext = train['comment_text'].fillna("_na_")
train_raw_text = train["comment_text"].fillna("_na_").values
test_raw_text = test["comment_text"].fillna("_na_").values
together = pd.concat([traintext,testtext]).values
tokenizer = text.Tokenizer(num_words=100000)
tokenizer.fit_on_texts(list(together))
train_tokenized = tokenizer.texts_to_sequences(train_raw_text)
test_tokenized = tokenizer.texts_to_sequences(test_raw_text)
x_train = sequence.pad_sequences(train_tokenized, maxlen=150)
x_test = sequence.pad_sequences(test_tokenized, maxlen=150)
y = train[classes].values
embed_size = 300 # how big is each word vector

embeddings_index = {}
f = open(EMBEDDING_FILE)
for line in f:
    values = line.split()
    word = ' '.join(values[:-300])
    coefs = np.asarray(values[-300:], dtype='float32')
    embeddings_index[word] = coefs.reshape(-1)
f.close()
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
word_index = tokenizer.word_index
nb_words = min(100000, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= 100000: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


model = Sequential()
model.add(Embedding(100000,embed_size,weights=[embedding_matrix]))
model.add(Bidirectional(LSTM(300, return_sequences=True, dropout=0.25, recurrent_dropout=0.25)))
model.add(GlobalMaxPool1D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(6, activation='sigmoid'))

#maybe augment data here?

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=2)

callbacks_list = [checkpoint, early]
model.fit(x_train,y, batch_size=32, epochs = 3, validation_split = 0.1, callbacks = callbacks_list, verbose=2)
model.load_weights(file_path)

y_test = model.predict([x_test],batch_size=1024, verbose=1)


sample_submission = pd.read_csv("/data/submission.csv")
sample_submission[classes] = y_test
sample_submission.to_csv("/output/try.csv", index=False)
