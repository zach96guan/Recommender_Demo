hive-init
# sudo su - athops
# hive
# quit;

# query sample data
/usr/local/bin/hive -e 'select * from ucid.common_transaction_model where source="wmt" and purchase_date="2019-05-20";' > /home/zach/sample_520.txt 

# query train/validation/test data
# two months
/usr/local/bin/hive -e 'select * from ucid.common_transaction_model where source="wmt" and purchase_date>="2019-03-25" and purchase_date<="2019-05-19";' > /home/zach/train.txt
# two weeks
/usr/local/bin/hive -e 'select * from ucid.common_transaction_model where source="wmt" and purchase_date>="2019-05-20" and purchase_date<="2019-06-02";' > /home/zach/validation.txt
# one week
/usr/local/bin/hive -e 'select * from ucid.common_transaction_model where source="wmt" and purchase_date>="2019-06-03" and purchase_date<="2019-06-09";' > /home/zach/test.txt 

# to get the number of lines of output, in comparison with result of query count(*)
wc -l train.txt  # 47938318
wc -l validation.txt  # 11546420
wc -l test.txt  # 6149672

# check file size
du -h train.txt  # 5.1G
du -h validation.txt  # 1.3G
du -h test.txt  # 666M

# transfer textfile to local
# scp z0g00mx@10.227.237.70:/home/zach/train.txt ./
