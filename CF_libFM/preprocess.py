"""
Preprocess queried dataset and clean the outliers.

Query command:
/usr/local/bin/hive -e 'select * from ucid.common_transaction_model where source="wmt" and purchase_date=...";'
"""

import os
import sys
import numpy as np
import json
from collections import defaultdict


if __name__ == '__main__':
	file_path = './data/'
	train_path = file_path + 'train.txt'
	test_path = file_path + 'test.txt'


	# first indexing for cid and catalog_item_id
	first_user_map, first_item_map = {}, {}
	user_id = item_id = 0

	with open(train_path, 'r') as f:
		for line in f.readlines():
			line = line.split()
			cid, catalog_item_id = line[0], line[4]

			if cid not in first_user_map:
				first_user_map[cid] = user_id
				user_id += 1
			if catalog_item_id != 'NULL' and catalog_item_id not in first_item_map:
				first_item_map[catalog_item_id] = item_id
				item_id += 1

	print("{} users, {} items in total".format(len(first_user_map), len(first_item_map)))


	# calculate frequencies
	user_dict, item_dict = defaultdict(int), defaultdict(int)

	with open(train_path, 'r') as f:
		for line in f.readlines():
			line = line.split()
			if line[4] == 'NULL':
				continue
			user_id, item_id = first_user_map[line[0]], first_item_map[line[4]]

			user_dict[user_id] += 1
			item_dict[item_id] += 1


	# set threshold to clean infrequent customers/items, resellers
	lower_user_threshold, upper_user_threshold, lower_item_threshold = 7, 350, 5
	ignored_user_set, ignored_item_set = set(), set()

	for user_id, purchase_times in user_dict.items():
		if purchase_times <= lower_user_threshold or purchase_times >= upper_user_threshold:
			ignored_user_set.add(user_id)

	for item_id, sale_times in item_dict.items():
		if sale_times <= lower_item_threshold:
			ignored_item_set.add(item_id)

	print("{} users, {} items to ignore".format(len(ignored_user_set), len(ignored_item_set)))


	# re-indexing after we clean the outliers
	train_user_map, train_item_map = {}, {}
	user_id = item_id = 0
	interaction = 0

	with open(train_path, 'r') as f:
		for line in f.readlines():
			line = line.split()
			cid, catalog_item_id = line[0], line[4]

			if catalog_item_id != 'NULL' and first_user_map[cid] not in ignored_user_set and \
			first_item_map[catalog_item_id] not in ignored_item_set:
				interaction += 1

				if cid not in train_user_map:
					train_user_map[cid] = user_id
					user_id += 1
				if catalog_item_id not in train_item_map:
					train_item_map[catalog_item_id] = item_id
					item_id += 1

	n_users, n_items = len(train_user_map), len(train_item_map)

	print("{} users, {} items after shrinking, density = {:.6f}%".format(n_users, n_items, 100*interaction/(n_users*n_items)))


	# write json mapping files
	cid_map, catalog_item_id_map = {}, {}

	cid_map = {v: k for k, v in train_user_map.items()}
	catalog_item_id_map = {v: k for k, v in train_item_map.items()}

	with open(file_path + 'cid2idx.json', 'w') as map1, open(file_path + 'itemid2idx.json', 'w') as map2, \
	open(file_path + 'idx2cid.json', 'w') as map3, open(file_path + 'idx2itemid.json', 'w') as map4:
		json.dump(train_user_map, map1)
		json.dump(train_item_map, map2)
		json.dump(cid_map, map3)
		json.dump(catalog_item_id_map, map4)


	# process train file to triple-rating format (tab-sep), with idx in ascending order,
	train_user_item_dict = defaultdict(lambda: defaultdict(int))

	with open(train_path, 'r') as f:
		for line in f.readlines():
			line = line.split()
			cid, catalog_item_id = line[0], line[4]

			if catalog_item_id != 'NULL' and cid in train_user_map and catalog_item_id in train_item_map:
				user_id, item_id = train_user_map[cid], train_item_map[catalog_item_id]
				train_user_item_dict[user_id][item_id] = 1

	with open(file_path + 'processed_train.txt', 'w') as f:
		for user_id in sorted(train_user_item_dict):
			for item_id in sorted(train_user_item_dict[user_id]):
				f.write("{}\t{}\t{}\n".format(user_id, item_id, train_user_item_dict[user_id][item_id]))


	# process test file to triple-rating format
	test_user_item_dict = defaultdict(lambda: defaultdict(int))

	with open(test_path, 'r') as f:
		for line in f.readlines():
			line = line.split()
			cid, catalog_item_id = line[0], line[4]

			if catalog_item_id != 'NULL' and cid in train_user_map and catalog_item_id in train_item_map:
				user_id, item_id = train_user_map[cid], train_item_map[catalog_item_id]
				test_user_item_dict[user_id][item_id] = 1

	with open(file_path + 'processed_test.txt', 'w') as f:
		for user_id in sorted(test_user_item_dict):
			for item_id in sorted(test_user_item_dict[user_id]):
				f.write("{}\t{}\t{}\n".format(user_id, item_id, test_user_item_dict[user_id][item_id]))


"""
Example output:

20596556 users, 3706433 items in total
18672730 users, 2831782 items to ignore
1856520 users, 831859 items after shrink, density = 0.001983%
"""


