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
from os.path import basename
import glob
import os
from PIL import Image
from pyspark import SparkContext
sc = SparkContext()
# To turn off unnecessary systemml warnings
sc.setLogLevel("ERROR")

def getLines(txtFile):
	with open(txtFile) as f:
		lines = f.read().splitlines()
	return lines


def canResize(filename):
	try:
		Image.open(filename).resize(224, 224)
		return True
	except:
		return False


def correctedLine(line):
	elem = line.split(' ')
	fileWithoutExtension = os.path.splitext(os.path.join(os.environ['IMAGENET_ROOT'], 'raw_images', elem[0]))[0]
	fileNames =  glob.glob(fileWithoutExtension + '.*')
	return fileNames[0] + ' ' + elem[1] and canResize(fileNames[0]) if len(fileNames) == 1 else 'INVALID'

	
def correctFile(trainFile):
	lines = sc.parallelize(getLines(trainFile)).map(lambda line : correctedLine(line)).collect()
	lines = [ line for line in lines if line != 'INVALID' ]
	with open(trainFile, "w") as f:
		for line in lines:
			f.write("%s\n" % line)


correctFile('train.txt')
correctFile('val.txt')

