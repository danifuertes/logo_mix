# LogoMix: A Data Augmentation Technique for Object Detection Applied to Logo Recognition

## Paper
Most of the public logo datasets are very small compared to other object detection datasets. This is a real problem for
Deep Learning based algorithms that require many samples to learn from. Here, we present LogoMix, a new data augmentation
method for object detection tasks and applied to logo recognition. For more details, please see our
[paper](https://ieeexplore.ieee.org/document/9730444). If this repository is useful for your work, please cite our
paper:

```
@INPROCEEDINGS{9730444,
    author={Fuertes, Daniel and del-Blanco, Carlos R. and Jaureguizar, Fernando and Giarcía, Narciso}, 
    booktitle={2022 IEEE International Conference on Consumer Electronics (ICCE)},
    title={LogoMix: A Data Augmentation Technique for Object Detection Applied to Logo Recognition},
    year={2022},
    pages={1-2},
    doi={10.1109/ICCE53296.2022.9730444}
}
``` 

## Software requirements

This code has been tested on Ubuntu 18.04.4 LTS with Python 3.8.5, CUDA 10.2 and a GPU TITAN Xp. The dependencies
can be obtained as follows:

1. Install CUDA 10.2 (other versions may be compatible) if not installed yet. Check
[Tesorflow website](https://www.tensorflow.org/install/gpu?hl=es-419) for additional information about GPU support.
2. Create a conda environment with `conda env create -f logo_detector.yml`. The version of Conda used is 4.10.3. Check
[Conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) to know how to install
Conda.
3. Activate conda environment with `conda activate logo_detector`. Deactivate the environment with `conda deactivate`.

## Pre-trained weights

You should download YoloV3 pre-trained weights from [Yolo website](https://pjreddie.com/darknet/yolo/) and convert them
to keras weights.

```bash
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights yolo3/yolo_weights.h5
```

## Dataset & Training

Your dataset should be divided on 2 main sets: train and test. The structure of your dataset should be similar to the
following one:
```
/DatasetName
    train.txt
    test.txt
    classes.txt
    /train
        /train_sequence_1
            000000000.png
            000000001.png
            000000002.png
            ...
        /train_sequence_2
          ...
        /train_sequence_N
          ...
    /test
        /test_sequence_1
            000000000.png
            000000001.png
            000000002.png
            ...
        /test_sequence_2
          ...
        /test_sequence_N
          ...
```
The file `classes.txt` should contain a list with all the classes of the dataset. The `train.txt` and `test.txt` files
should contain a list of the train and test annotations, respectively. The ground-truth format is described next:
```
relative/path/to/train_sequence_1/000000000.png left,top,right,bottom,class
relative/path/to/train_sequence_1/000000001.png left,top,right,bottom,class left,top,right,bottom,class
relative/path/to/train_sequence_1/000000002.png
relative/path/to/train_sequence_1/000000000.png left,top,right,bottom,class
...
relative/path/to/train_sequence_2/000000000.png left,top,right,bottom,class
relative/path/to/train_sequence_2/000000001.png
relative/path/to/train_sequence_2/000000002.png left,top,right,bottom,class left,top,right,bottom,class left,top,right,bottom,class
...
```
where x and y are the coordinates of each of the point-based annotations on the image. You can configure your validation
data like your train and test data. The file with the annotations of your validation set should be called `val.txt`. If
your dataset does not contain a validation set, you can provide a percentage with the option `--val_perc` to extract
some random samples from the training set and use them to validate:

```bash
python train.py --dataset_path /path/to/directory/containing/your/dataset --dataset_name DatasetName --val_perc 0.1 --img_width 448 --img_height 448
```

In case you have a validation set with a format similar to the one described above, you can train your model with:

```bash
python train.py --dataset_path /path/to/directory/containing/your/dataset --dataset_name DatasetName --img_width 448 --img_height 448
```

While you are training a model, the weights that optimize the validation loss are saved in 
`models/YoloV3_DatasetName_CurrentDate` by default. To restore a model, you should use the option
`--restore_model True`, indicate the path to the model with `--save_dir models/model_DatasetName_TrainDate` and indicate
the weights desired with `--weights weights_057.h5`. Example:

```bash
python train.py --restore_model True --save_dir models/YoloV3_DatasetName_TrainDate --weights weights_057.h5 --dataset_path /path/to/directory/containing/your/dataset --dataset_name DatasetName --img_width 448 --img_height 448
```
You can use the option `--new_anchors True` to calculate new anchors for YoloV3 based on your data. For any additional
help, you can run:

```bash
python train.py --help
```

#  Test

To evaluate your trained model using your test data with the format described above, you can run:

```bash
python test.py --save_dir models/YoloV3_DatasetName_TrainDate --weights weights_057.h5 --dataset_path /path/to/directory/containing/your/dataset --dataset_name DatasetName --img_width 448 --img_height 448
```

Note that options related to the structure of the network should not be changed. In case you do not remember any of the
options, read the file `models/model_DatasetName_TrainDate/log_dir/options.txt`, that contains a list with the options
used to train that model.

To visualize the detections, you can use the option `--show_pred True`.

```bash
python test.py --show_pred True --save_dir models/YoloV3_DatasetName_TrainDate --weights weights_057.h5 --dataset_path /path/to/directory/containing/your/dataset --dataset_name DatasetName --img_width 448 --img_height 448
```

You can save the images with the predictions by running:

```bash
python test.py --save_image True --save_dir models/YoloV3_DatasetName_TrainDate --weights weights_057.h5 --dataset_path /path/to/directory/containing/your/dataset --dataset_name DatasetName --img_width 448 --img_height 448
```

The images are saved by sequences in `map/predictions/YoloV3_DatasetName_TrainDate_test_TestDate/images`. Next to this
directory, you can find 2 directories called `detection-results` and `ground-truth`. These directories contain files
with the predictions and annotations of each test image, respectively. To evaluate your model with metrics like
Precision (P), Recall (R), F1-Score (F), mean Average Precision (mAP), Miss Rate (MR), False Positives Per Images
(FPPI), and Log Average Miss Rate (LAMR), it is necessary to run another script:

```bash
python -m map.map --results_dir map/predictions/YoloV3_DatasetName_TrainDate_test_TestDate --img_width 448 --img_height 448
```

Check the folder `map/predictions/model_DatasetName_TrainDate_test_TestDate/results` to find the results computed. For
any additional help, you can run:

```bash
python test.py --help
python -m map.map --help
```

## Acknowledgements
This repository is an adaptation of [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3) for the case of logo
detection + LogoMix.
