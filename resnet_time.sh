#!/bin/bash

# Runs ResNet with caffe on different minibatch size and generates time.txt in $CAFFE_ROOT folder
# Expects CAFFE_ROOT to be set. Also, please store imagenet dataset in examples/imagenet/ilsvrc12_train_lmdb 

if [[ -z "${CAFFE_ROOT}" ]]; then
	echo 'The environment variable CAFFE_ROOT needs to be defined.'
	exit 1
fi
cd $CAFFE_ROOT
echo "" > time.txt
for NUM_ITER in 10 100
do
for batch_size in 32 64 128 256 512 1024
do
	wget https://raw.githubusercontent.com/niketanpansare/dl-benchmark/master/ResNet_50_network.proto
	sed -i "12s/.*/    batch_size: "$batch_size"/" ResNet_50_network.proto
	if [ "$batch_size" == "32" ]
	then
		echo "------------------------------------------------" >> time.txt
		echo "With single GPU and minibatch size="$batch_size" and iterations="$NUM_ITER >> time.txt
		$CAFFE_ROOT/build/tools/caffe time --model=ResNet_50_network.proto --iterations=$NUM_ITER --gpu 0 | grep " ms\.$" >> time.txt
	fi
	
	echo "------------------------------------------------" >> time.txt
	echo "With single GPU and minibatch size="$batch_size" and iterations="$NUM_ITER >> time.txt
	$CAFFE_ROOT/build/tools/caffe time --model=ResNet_50_network.proto --iterations=$NUM_ITER | grep " ms\.$" >> time.txt
done
done
