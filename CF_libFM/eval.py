"""
Evaluate on ranking metrics performance after we run the model by bash script.
"""
import os
import sys
import json
import numpy as np 
import pandas as pd 

from collections import defaultdict
from scipy.sparse import csr_matrix
from itertools import islice
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors


## Define ranking based metrics
# mean average precision
def AP_k(y_label, y_pred, k):
    ground_truth = set(y_label)
    if len(ground_truth) == 0: return 0
    
    hit_cnt, precision = 0, 0.0
    for i, p in enumerate(y_pred):
        if p in ground_truth:
            hit_cnt += 1
            precision += hit_cnt / (i + 1.0)

    precision_k = precision / min(len(ground_truth), k)
    return precision_k

def MAP_k(model, ratings, k):
    precision = 0
    n_users = ratings.shape[0]
    
    for user in range(n_users):
        y_label = ratings[user].indices
        u_pred = model._predict_user(user)
        y_pred = np.argsort(u_pred)[::-1][:k]
        precision += AP_k(y_label, y_pred, k)
    
    precision /= n_users
    return precision


# hit ratio
def HR_k_helper(y_label, y_pred, k):
    ground_truth = set(y_label)
    if len(ground_truth) == 0: return 0, 0
    
    hit_cnt = 0
    for p in y_pred:
        if p in ground_truth:
            hit_cnt += 1
    return hit_cnt, len(ground_truth)

def HR_k(model, ratings, k):
    hits, actuals = 0, 0
    n_users = ratings.shape[0]
    
    for user in range(n_users):
        y_label = ratings[user].indices
        u_pred = model._predict_user(user)
        y_pred = np.argsort(u_pred)[::-1][:k]
        
        tmp1, tmp2 = HR_k_helper(y_label, y_pred, k)
        hits += tmp1
        actuals += tmp2
        
    return hits / actuals


# normalized discounted cummulative gain
def DCG_k(y_label, y_pred, k):
    ranked = np.argsort(y_pred)[::-1]
    y_label = np.take(y_label, ranked[:k])

    gains = 2 ** y_label - 1
    discounts = np.log2(np.arange(2, gains.size + 2))

    result = np.sum(gains / discounts)
    return result

def NDCG_k_helper(y_label, y_pred, k):
    actual = DCG_k(y_label, y_pred, k)
    best = DCG_k(y_label, y_label, k)
    
    if not best:
        return 0
    return actual / best

def NDCG_k(model, ratings, k):
    result = 0.0
    n_users, n_items = ratings.shape
    
    for user in range(n_users):
        y_label = np.zeros(n_items)
        indices = ratings[user].indices
        y_label[indices] = 1
        u_pred = model._predict_user(user)
        result += NDCG_k_helper(y_label, u_pred, k)
    
    result /= n_users
    return result



## Define class for convenience of recommendation
class FM:
    def __init__(self, user_factors, item_factors, n_users, n_items):
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.n_users = n_users
        self.n_items = n_items
    
    def predict(self):
        prediction = self.user_factors.dot(self.item_factors.T)
        return prediction
    
    def _predict_user(self, user):
        u_pred = self.user_factors[user].dot(self.item_factors.T)
        return u_pred
    
    # return top-N ranked items for given users
    def recommend(self, users_list, N=5):
        length = len(users_list)
        recommendation = np.zeros((length, N), dtype=np.uint32)
        
        for i, uid in enumerate(users_list):
            recommendation[i] = self._recommend_user(uid, N)
        return recommendation
    
    def _recommend_user(self, user, N=5):
        u_pred = self._predict_user(user)
        
        indices = np.argpartition(u_pred, -N)[-N:]
        ptrs = np.argsort(u_pred[indices])[::-1]
        ranked = indices[ptrs]
        
        ret = list(islice((rec for rec in ranked), N))
        return ret

     # return top-N similar items for given items
    def _similar_items(self, item_idxs, N=5):
        normed = normalize(self.item_factors)
        
        knn = NearestNeighbors(n_neighbors=N+1, metric='euclidean')
        knn.fit(normed)
        
        selected = normed[item_idxs]
        _, ret = knn.kneighbors(selected)
        return ret[:, 1:]



## Convert raw test data to sparse matrix
def create_test_matrix(data, user_col, item_col, rating_col, N, M):
    """
    rtype: scipy sparse matrix csr_matrix, shape same as train.txt
    """
    rows = np.array(data[user_col])
    cols = np.array(data[item_col])
    ratings = np.array(data[rating_col])
    
    result = csr_matrix((ratings, (rows, cols)), shape=(N, M))
    result.eliminate_zeros()
    return result, data



if __name__ == '__main__':
	file_path = './output/'
	names = ['userId', 'itemId', 'rating']

	with open('./data/cid2idx.json', 'r') as map1, open('./data/itemid2idx.json', 'r') as map2:
		train_user_map, train_item_map = json.load(map1), json.load(map2)

	n_users, n_items = len(train_user_map), len(train_item_map)


	# read test file and convert to csr_matrix format
	test_path = './data/processed_test.txt'
	df_t = pd.read_csv(test_path, sep='\t', header=None, names=names)
	X_test, df_t = create_test_matrix(df_t, names[0], names[1], names[2], n_users, n_items)


	# get latent factors from .model file
	model_name = 'fm_dim_1,1,8_iter_20_method_als_reg_0,0,10_std_0.1.model'
	factors = []

	with open(file_path + model_name, 'r') as f:
		flag = False   # mark the place to read
		for line in f.readlines():
			if line.startswith("#pairwise interactions"):
				flag = True 
				continue
			if flag:
				factors.append(list(map(float, line.split())))

	# sanity check
	assert len(factors) == (n_users + n_items)

	# save latent factors
	user_factors, item_factors = np.array(factors[:n_users]), np.array(factors[n_users:])
	np.save(file_path + 'user_factors.npy', user_factors)
	np.save(file_path + 'item_factors.npy', item_factors)


	# recommendation if given user or customer
	fm = FM(user_factors, item_factors, n_users, n_items)


"""
Example:

k = 20
print("MAP@{}: {}".format(k, MAP_k(fm, X_test, k)))
print("NDCG@{}: {}".format(k, NDCG_k(fm, X_test, k)))
print("HR@{}: {}".format(k, HR_k(fm, X_test, k)))

n_samples = 10
sampled_users = np.random.choice(list(train_user_map.keys()), n_samples)
sampled_user_idxs = list(map(lambda x:train_user_map.get(x), sampled_users))
print(fm.recommend(sampled_user_idxs, N=5))

sampled_items = np.random.choice(list(train_item_map.keys()), n_samples)
sampled_item_idxs = list(map(lambda x:train_item_map.get(x), sampled_items))
print(fm._similar_items(sampled_item_idxs, N=5))
"""


