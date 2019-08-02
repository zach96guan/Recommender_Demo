# Personalized Recommender for Implicit Data
In this project, the target is to design a personalized recommendation system for implicit feedback data. I implement latent factor model based collaborative filtering, including pointwise/pairwise non-negative matrix factorization and factorization machine methods. To evaluate the proposed model, I perform the experiment in following steps. On one hand, to capture sequential users’ purchase behavior, split into the train/validation data in temporal manner. On the other hand, as items ranked higher should be more preferred by customers, ranking based metrics are applied. The experiment with Walmart.com transaction data shows potential improvement on top-N recommendation performance.

## Data
- Database: `ucid`
- Table: `common_transaction_model`
- Columns: `wmt`, `cid`, `catalog_item_id`
- Features: implicit data with sparsity and imbalance

## Target
- Users' response based top-N recommendation
- Personalized ranking

## Model: latent factor model based collaborative filtering
- Non-negative matrix factorization
	- Pointwise, alternating least square with regularization (*ALS_WR*)
	- Pairwise, Bayesian personalized ranking (*BPR*)
- Factorization machine with libFM

## Ranking Metrics
1. Hit ratio (*HR*)
2. Normalized discounted cumulative gain (*NDCG*)
3. Mean average precision (*MAP*)
4. Area under curve (*AUC*)

## Experiment
- `Train.txt`: two months, from ‘2019-03-25’ to ‘2019-05-19’
- `Validation.txt`: two weeks, from ‘2019-05-20’ to ‘2019-06-02’
- `Test.txt`: one week, from ‘2019-06-03’ to ‘2019-06-09’
