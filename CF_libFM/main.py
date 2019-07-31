"""
Get standard libFM input for matrix factorization.
"""

import os
import sys
import json 
import numpy as np 

from numba import jit
from collections import defaultdict


# negative sampling from non-observed data to decrease imbalance
@jit(nopython=True)
def build_negative_sample_table(item_count_scale, table_size):
	# table size should be larger than n_items
	neg_sample_table = []
	z = np.sum(item_count_scale)

	for i, v in enumerate(item_count_scale):
		j = 0
		while j < int(table_size * v / z):
			neg_sample_table.append(i)
			j += 1

	return neg_sample_table


# assume 1:1 sample given explicit data
@jit(nopython=True)
def batch_sample(neg_sample_table, pos):
	sampled = neg_sample_table[pos]
	pos = (pos + 1) % len(neg_sample_table)
	return sampled, pos


class Neg_sampler:
	def __init__(self, item_count, scale=0.5):
		self.TABLE_SIZE = 3e6
		self.item_count_scale = (np.array(item_count) ** scale)
		self.pos = 0

	def build(self):
		self.neg_sample_table = np.array(build_negative_sample_table(self.item_count_scale, self.TABLE_SIZE))
		np.random.shuffle(self.neg_sample_table)

	def sample(self):
		idx, self.pos = batch_sample(self.neg_sample_table, self.pos)
		return idx


if __name__ == '__main__':
	file_path = './data/'

	with open('./data/cid2idx.json', 'r') as map1, open('./data/itemid2idx.json', 'r') as map2:
		train_user_map, train_item_map = json.load(map1), json.load(map2)

	n_users, n_items = len(train_user_map), len(train_item_map)


	# negative sample
	train_purchase_dict = defaultdict(set)   # record observed data
	item_count = np.zeros(n_items)   # for weighted information

	with open(file_path + 'processed_train.txt', 'r') as t1:
		for line in t1.readlines():
			user_id, item_id, rating = line.split('\t')
			item_count[int(item_id)] += 1
			train_purchase_dict[user_id].add(item_id)

	sampler = Neg_sampler(item_count)
	sampler.build()


	# write train file
	with open(file_path + 'processed_train.txt', 'r') as f1, open(file_path + 'libfm_train.txt', 'w') as g1:
		for line in f1.readlines():
			user_id, item_id, rating = line.split('\t')
			g1.write("1 {}:1 {}:1\n".format(user_id, str(n_users + int(item_id))))

			while True:
				neg_idx = sampler.sample()
				if str(neg_idx) not in train_purchase_dict[user_id]: 
					break
			g1.write("-1 {}:1 {}:1\n".format(user_id, str(n_users + int(neg_idx))))


	# write test file, consider mandatory option
	test_size = 0.05
	test_purchase_dict = defaultdict(set)

	with open(file_path + 'processed_test.txt', 'r') as t2:
		for line in t2.readlines():
			user_id, item_id, rating = line.split('\t')
			test_purchase_dict[user_id].add(item_id)

	with open(file_path + 'processed_test.txt', 'r') as f2, open(file_path + 'libfm_test.txt', 'w') as g2:
		for line in f2.readlines():
			if np.random.uniform(0, 1, 1)[0] >= test_size:
				continue

			user_id, item_id, rating = line.split('\t')
			g2.write("1 {}:1 {}:1\n".format(user_id, str(n_users + int(item_id))))

			while True:
				neg_idx = sampler.sample()
				if str(neg_idx) not in test_purchase_dict[user_id]: 
					break
			g2.write("-1 {}:1 {}:1\n".format(user_id, str(n_users + int(neg_idx))))


# if don't sample, we can convert to recommender files directly by given perl script
# ./libFM/scripts/triple_format_to_libfm.pl -in ../processed_train.txt,../processed_test.txt -target 2 -separator "\t"


