#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

# Usage:
# For classification problems,
# $SPARK_HOME/bin/spark-submit --driver-memory 30g convert_binaryblocks_lmdb.py X.train.mtx Y.train.mtx train.lmdb 64 $NUM_CHANNELS $HEIGHT $WIDTH
#
# For image segmentation / multi-label problems, generate two lmdbs:
# $SPARK_HOME/bin/spark-submit --driver-memory 30g convert_binaryblocks_lmdb.py X.train.mtx None X.train.lmdb 64 $NUM_CHANNELS $HEIGHT $WIDTH
# $SPARK_HOME/bin/spark-submit --driver-memory 30g convert_binaryblocks_lmdb.py Y.train.mtx None Y.train.lmdb 64 $NUM_LABELS 1 1
import lmdb, caffe
import numpy as np
import sys
input_file_x = sys.argv[1]
# Use input_file_y == 'None' for image segmentation problem
input_file_y = sys.argv[2]
output_lmdb_file = sys.argv[3]
BUFFER_SIZE = int(sys.argv[4])
num_channels = int(sys.argv[5])
height = int(sys.argv[6])
width = int(sys.argv[7])

from pyspark import SparkContext
sc = SparkContext()
# To turn off unnecessary systemml warnings
sc.setLogLevel("ERROR")
from systemml import MLContext, dml
ml = MLContext(sc)

# create the lmdb file
lmdb_env = lmdb.open(output_lmdb_file, map_size=int(1e12))
lmdb_txn = lmdb_env.begin(write=True)
datum = caffe.proto.caffe_pb2.Datum()

start_index = 1
xlen = float(ml.execute(dml('X = read("' + input_file_x + '"); xlen = nrow(X)').output('xlen')).get('xlen'))
print('Number of data items:' + str(xlen))
while start_index < xlen:
	end_index = start_index + BUFFER_SIZE
	if end_index > xlen:
		end_index = xlen	
	dmlStrX = 'X = read("' + input_file_x + '"); X = X[' + str(start_index) + ':' + str(end_index) + ',]'
	X = ml.execute(dml(dmlStrX).output('X')).get('X').toNumPy()
	X = X.reshape((-1, num_channels, height, width))
	batch_size = X.shape[0]
	if input_file_y != 'None':
		dmlStrY = 'y = read("' + input_file_y + '"); y = y[' + str(start_index) + ':' + str(end_index) + ',]'
		y = ml.execute(dml(dmlStrY).output('y')).get('y').toNumPy()
		y = y.reshape((batch_size, -1))
	else:
		y = None
	for i in range(batch_size):
		datum = caffe.io.array_to_datum(X[i], y[i]) if y is not None else caffe.io.array_to_datum(X[i])
		keystr = '{:0>8d}'.format(i+start_index)
		lmdb_txn.put( keystr, datum.SerializeToString() )
	lmdb_txn.commit()
	lmdb_txn = lmdb_env.begin(write=True)
	start_index = start_index + BUFFER_SIZE
