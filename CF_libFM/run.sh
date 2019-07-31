## use libFM to get latent factors
#!/bin/bash
# cd ./CF_libFM

## set parameters
if [[ $# -ge 1 ]]
then
	dim=$1
else
	dim='1,1,8'
fi

if [[ $# -ge 2 ]]
then
	iter=$2
else
	iter=20
fi

if [[ $# -ge 3 ]]
then
	method=$3
else
	method='als'
fi

if [[ $# -ge 4 ]]
then
	reg=$4
else
	reg='0,0,10'
fi

if [[ $# -ge 5 ]]
then
	std=$5
else
	std=0.1
fi


model_name="./output/fm_dim_${dim}_iter_${iter}_method_${method}_reg_${reg}_std_${std}"

## run, nohup if necessary
./libfm/bin/libFM -task c -train ./data/libfm_train.txt -test ./data/libfm_test.txt \
-dim $dim -iter $iter -method $method -regular $reg -init_stdev $std \
-out "${model_name}.out" -rlog "${model_name}.rlog" -save_model "${model_name}.model" -verbosity 1
# > "./output/output.out" &


