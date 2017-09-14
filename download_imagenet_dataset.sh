#!/bin/bash

if [[ -z "${IMAGENET_ROOT}" ]]; then
	echo 'The environment variable IMAGENET_ROOT needs to be defined.'
	exit 1
fi

cd $IMAGENET_ROOT
# First download the URLs to images of imagenet dataset:
wget http://www.deepdetect.com/dd/datasets/imagenet/ilsvrc12_urls.txt.gz
gzip -d ilsvrc12_urls.txt.gz

# Download the images mentioned in the URLs. This would take lot of time, so you may want to use only subset of URLs in ilsvrc12_urls.txt:
mkdir raw_images
cd $IMAGENET_ROOT/raw_images
wget https://raw.githubusercontent.com/beniz/imagenet_downloader/master/download_imagenet_dataset.py
python download_imagenet_dataset.py ../ilsvrc12_urls.txt . --jobs 100 --retry 3 --sleep 0 &> /dev/null
rm download_imagenet_dataset.py
echo 'Successfully downloaded the imagenet dataset: '$IMAGENET_ROOT'/raw_images'
