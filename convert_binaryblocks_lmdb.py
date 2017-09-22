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

import lmdb, caffe
import numpy as np
import sys
input_file_x = sys.argv[1]
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
ylen = float(ml.execute(dml('y = read(' + input_file_y + '); ylen = nrow(y)').output('ylen')).get('ylen'))
print('Number of data items:' + str(ylen))
while start_index < ylen:
	end_index = start_index + BUFFER_SIZE
	if end_index > ylen:
		end_index = ylen	
	dmlStrX = 'X = read(' + input_file_x + '); X = X[' + str(start_index) + ':' + str(end_index) + ',]'
	dmlStrY = 'y = read(' + input_file_y + '); y = y[' + str(start_index) + ':' + str(end_index) + ',]'
	X = ml.execute(dml(dmlStrX).output('X')).get('X').toNumPy()
	y = ml.execute(dml(dmlStrX).output('y')).get('y').toNumPy()
	X = X.reshape((-1, num_channels, height, width))
	batch_size = X.shape[0]
	y = y.reshape((batch_size, -1))
	for i in range(batch_size):
		datum = caffe.io.array_to_datum(X[i], y[i])
		keystr = '{:0>8d}'.format(i+start_index)
		lmdb_txn.put( keystr, datum.SerializeToString() )
	lmdb_txn.commit()
	lmdb_txn = lmdb_env.begin(write=True)
	start_index = start_index + BUFFER_SIZE
