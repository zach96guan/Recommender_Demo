hive> describe formatted common_transaction_model;

hive> set hive.cli.print.header=true;

hive> select * from ucid.common_transaction_model where source='wmt' and purchase_date='2019-05-20' order by common_transaction_model.retail_price desc limit 10;

hive> select count(*), avg(retail_price), variance(retail_price) from ucid.common_transaction_model where source='wmt' and purchase_date='2019-05-20';