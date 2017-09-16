- Find all downloaded files:

```bash
CURR_DIR=`pwd`
export IMAGENET_ROOT=/home/npansare/imagenet
cd $IMAGENET_ROOT/raw_images
find -follow > $CURR_DIR/all_files.txt
cd $CURR_DIR
ls $IMAGENET_ROOT/raw_images > all_dir.txt
sort all_dir.txt -o all_dir.txt
sort all_files.txt -o all_files.txt
comm -13 all_dir.txt all_files.txt > all_files1.txt
```

```python
import pandas as pd
p1 = pd.read_csv('all_files1.txt', names=['filename', 'suffix'], sep=' ')
p2 = pd.read_csv('caffe_train.txt', names=['filename', 'label'], sep=' ')
p3 = p1.set_index('filename').join(p2.set_index('filename'))
import numpy as np
df = p3[np.isfinite(p3['label'])]
df = df.reset_index()
df['path'] = df[['filename', 'suffix']].apply(lambda x: '.'.join(x), axis=1)
df = df.drop('filename', 1)
df = df.drop('suffix', 1)
df1 = df[['path', 'label']]
df1.to_csv('downloaded_caffe_train.txt', sep=' ', index=False)
```

- Generate lmdb 

```bash
export CAFFE_ROOT=/home/npansare/nike/caffe
export IMAGENET_ROOT=/home/npansare/imagenet
nohup $CAFFE_ROOT/build/tools/convert_imageset --resize_height=224 --resize_width=224 --shuffle $IMAGENET_ROOT/raw_images downloaded_caffe_train.txt  $CAFFE_ROOT/examples/imagenet/ilsvrc12_train_lmdb &> caffe_lmdb_log.txt &
```

- Remove images that failed resizing:

```bash
grep io.cpp caffe_lmdb_log.txt | rev | cut -d' ' -f1 | rev > failed_resize.txt
```

```python
import pandas as pd
with open('failed_resize.txt') as f:
	lines = f.read().splitlines()

df = pd.read_csv('downloaded_caffe_train.txt', names=['filename', 'label'], sep=' ')
df1 = df[~df['filename'].isin(lines)]
df1.to_csv('train_imagenet_1k.txt', sep=' ', index=False)
```
