# Store

In this module, we provide a class named SampleStore that stores
many basic reader after parsing the list life which lists all
the images and the corresponding ground-truth.

### List file (.lst) for a regular semantic segmentation task

For the task of regular semantic segmentation, the list file can
be seen as follows.
```angular2html
irrg/top_mosaic_09cm_area1.tif gt/top_mosaic_09cm_area1.tif
irrg/top_mosaic_09cm_area13.tif gt/top_mosaic_09cm_area13.tif
irrg/top_mosaic_09cm_area17.tif gt/top_mosaic_09cm_area17.tif
irrg/top_mosaic_09cm_area21.tif gt/top_mosaic_09cm_area21.tif
irrg/top_mosaic_09cm_area23.tif gt/top_mosaic_09cm_area23.tif
irrg/top_mosaic_09cm_area26.tif gt/top_mosaic_09cm_area26.tif
irrg/top_mosaic_09cm_area3.tif gt/top_mosaic_09cm_area3.tif
irrg/top_mosaic_09cm_area32.tif gt/top_mosaic_09cm_area32.tif
irrg/top_mosaic_09cm_area37.tif gt/top_mosaic_09cm_area37.tif
irrg/top_mosaic_09cm_area5.tif gt/top_mosaic_09cm_area5.tif
irrg/top_mosaic_09cm_area7.tif gt/top_mosaic_09cm_area7.tif
```

The first column refers to the true orthophoto and the second
column refers to the ground truth. If the ground truth is not provided,
the second column can be deleted.

### List file (.lst) for semantic segmentation based on multi-modal data
For the task of semantic segmentation based on multi-modal data,
the list file can be seen as follows.

```angular2html
irrg/top_mosaic_09cm_area1.tif dsm/dsm_09cm_matching_area1.tif gt/top_mosaic_09cm_area1.tif
irrg/top_mosaic_09cm_area13.tif dsm/dsm_09cm_matching_area13.tif gt/top_mosaic_09cm_area13.tif
irrg/top_mosaic_09cm_area17.tif dsm/dsm_09cm_matching_area17.tif gt/top_mosaic_09cm_area17.tif
irrg/top_mosaic_09cm_area21.tif dsm/dsm_09cm_matching_area21.tif gt/top_mosaic_09cm_area21.tif
irrg/top_mosaic_09cm_area23.tif dsm/dsm_09cm_matching_area23.tif gt/top_mosaic_09cm_area23.tif
irrg/top_mosaic_09cm_area26.tif dsm/dsm_09cm_matching_area26.tif gt/top_mosaic_09cm_area26.tif
irrg/top_mosaic_09cm_area3.tif dsm/dsm_09cm_matching_area3.tif gt/top_mosaic_09cm_area3.tif
irrg/top_mosaic_09cm_area32.tif dsm/dsm_09cm_matching_area32.tif gt/top_mosaic_09cm_area32.tif
irrg/top_mosaic_09cm_area37.tif dsm/dsm_09cm_matching_area37.tif gt/top_mosaic_09cm_area37.tif
irrg/top_mosaic_09cm_area5.tif dsm/dsm_09cm_matching_area5.tif gt/top_mosaic_09cm_area5.tif
irrg/top_mosaic_09cm_area7.tif dsm/dsm_09cm_matching_area7.tif gt/top_mosaic_09cm_area7.tif
```
The first column refers to the true orthophoto, 
the second refers to the Digital Surface Model (DSM),
and the third column refers to the ground truth.
If the ground truth is not provided,
the last column can be deleted.

### ComParser
This class parses each line of the .lst file and assigns
one basic reader to each image file.

#### Extension
The users can define their own .lst file and the corresponding
parser class if only it provides the same api as ComParser to
be called by the SampleStore.

### SampleStore
It assigns one basic reader for each image file
and keeps all the readers in this class.
In theory, one line in the .lst file can be seen as a sample
and one samples consists of many readers.

```angular2html
store = SampleStore(lstfile='/path/to/the/list_file/list_file.lst',
                    data_root='/path/to/the/data_root', withgt=True, openfirstly=True,
                    lvreadertype='gdal', parsertype=ComParser)
```
or 
```angular2html
store = SampleStore(lstfile='/path/to/the/list_file/list_file.lst',
                    data_root='/path/to/the/data_root', withgt=True, openfirstly=True,
                    lvreadertype='gdal', parsertype='common')
```

Users can get the samples and reader like this:
```angular2html
sample = store.getOneSample(2) # get the second sample
sample = store.getOneSample(-1) # get one sample by random
samples = store.getSomeSamples(10) #get 10 samples by random
topreaders = sample.top
gtreaders = sample.gt
nchannel = sample.nchannel
```

---------------
## Test
The test instance can be seen from __test/store.ipynb__
