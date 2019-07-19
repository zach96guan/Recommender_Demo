# Recommender_Demo
Intern project to implement recommender demo for implicit feedback transaction datasets.

## Data
- Database: `ucid`
- Table: `common_transaction_model`

## Columns
- `common_transaction_model.cid`
- `common_transaction_model.catalog_item_id`

## Features
- Implicit dataset
- Sparsity and imbalance

## Goal
- User response based recommendation
- Personalized ranking

## Experiment
- `Train.txt`: two months, from ‘2019-03-25’ to ‘2019-05-19’
- `Validation.txt`: two weeks, from ‘2019-05-20’ to ‘2019-06-02’
- `Test.txt`: one week, from ‘2019-06-03’ to ‘2019-06-09’

## Model
- ALS_WR based Collaborative Filtering
- Bayesian Personalized Ranking
