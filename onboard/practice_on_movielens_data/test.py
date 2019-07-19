import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

from sklearn.utils import shuffle
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz

import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'


"""
preprocess
"""
# df = pd.read_csv('rating.csv')

# df.userId -= 1
# unique_movie_ids = set(df.movieId.values)

# movie2idx = {}
# count = 0 

# for movie_id in unique_movie_ids:
# 	movie2idx[movie_id] = count
# 	count += 1

# df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)
# df = df.drop(columns=['timestamp'])

# df.to_csv('edited_rating.csv', index=False)


# """
# preprocess2sparse
# """
df = pd.read_csv('edited_rating.csv')

N = df.userId.max() + 1
M = df.movie_idx.max() + 1

df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

# A = lil_matrix((N, M))
# cnt = 0

# def update_train(row):
# 	global cnt
# 	cnt += 1
# 	i, j = int(row.userId), int(row.movie_idx)
# 	A[i, j] = row.rating

# df_train.apply(update_train, axis=1)


# A = A.tocsr()
# mask = (A > 0)
# save_npz('Atrain.npz', A)


# A_test = lil_matrix((N, M))
# cnt2 = 0

# def update_test(row):
# 	global cnt2
# 	cnt += 1
# 	i, j = int(row.userId), int(row.movie_idx)
# 	A_test[i, j] = row.rating

# df_test.apply(update_test, axis=1)

# A_test = A_test.tocsr()
# mask_test = (A_test > 0)
# save_npz('Atest.npz', A_test)


"""
use keras for matrix-factorization
"""
# initialize variables
K = 10  # latent dimensionality
mu = df_train.rating.mean()
epochs = 10
reg = 0.  # regularization penalty

u = Input(shape=(1, ))
m = Input(shape=(1, ))
u_embed = Embedding(N, K, embeddings_regularizer=l2(reg))(u) # (N, 1, K)
m_embed = Embedding(M, K, embeddings_regularizer=l2(reg))(m) # (N, 1, K)

u_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(u) # (N, 1, 1)
m_bias = Embedding(M, 1, embeddings_regularizer=l2(reg))(m) # (N, 1, 1)

x = Dot(axes=2)([u_embed, m_embed]) # (N, 1, 1)
x = Add()([x, u_bias, m_bias])
x = Flatten()(x) # (N, 1)


model = Model(inputs=[u, m], outputs=x)
model.compile(
  loss='mse',
  # optimizer='adam',
  # optimizer=Adam(lr=0.01),
  optimizer=SGD(lr=0.08, momentum=0.9),
  metrics=['mse'],
)

r = model.fit(
  x=[df_train.userId.values, df_train.movie_idx.values],
  y=df_train.rating.values - mu,
  epochs=epochs,
  batch_size=128,
  validation_data=(
    [df_test.userId.values, df_test.movie_idx.values],
    df_test.rating.values - mu
  )
)


# plot losses
plt.plot(r.history['loss'], label="train loss")
plt.plot(r.history['val_loss'], label="test loss")
plt.legend()
plt.show()

# plot mse
plt.plot(r.history['mean_squared_error'], label="train mse")
plt.plot(r.history['val_mean_squared_error'], label="test mse")
plt.legend()
plt.show()

