### Train Feature Extraction Model

The feature extraction model is a pixel-wise segmentation model of the
FCN-8s standard architecture, trained
on [mit sceneparsing benchmark](http://sceneparsing.csail.mit.edu/)
dataset.

This repo is forked from
https://github.com/meetshah1995/pytorch-semseg , which contains the
implementation of FCN-8s model.

1. Download
   ["Secene Parsing" dataset](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) from
   http://sceneparsing.csail.mit.edu/ and extract to hard disk.

1. Edit "path" in `configs/fcn8s_deepnav.yml` to refect the path to the training dataset.

1. Install Python package dependencies (please use Python 3, recommend to use [Miniconda](https://conda.io/miniconda.html) as the virtual Python environment).
   ```
   # steps to setup the conda Python virtial environment, can be skipped
   $ conda create --name deepnav python=3.5
   $ conda activate deepnav

   $ pip install -r requirements.txt
   ```
   
1. Start training
   ```
   $ python train.py --config configs/fcn8s_deepnav.yml
   ```

You will find running logs and model checkpoint in
`runs/fcn8s_deepnav/$PID`. The model checkpoint will be used in the
latter steps.

### Infer Feature Extraction Model

```
# save the segmented images to "out_path"
$ python infer.py --model_path fcn8s_mit_sceneparsing_benchmark_best_model.pkl --img_path 1539287494.103636.jpg  --out_path a.jpg
```

### Train Classifier Model

The classifier makes prediction based on the output of the feature
extraction model. The weights of the feature extraction model is
freezed during the classifier model training.

Use the command below to train the classifier model:

```
# feature_model_path: path to the saved weights of the feature extraction model.
# classifier_model_path: path to the saved weights of the classifier model. Use this argument when resuming training.
# train_csv_path: path to the csv training data.
#  An example train.csv:
#  image,label
#  IMG_9381.JPG,0
#  IMG_9382.JPG,0
#  IMG_9383.JPG,1
#  IMG_9384.JPG,1
#  IMG_9385.JPG,2
#  IMG_9386.JPG,2
#  The image colome contains the relative path to the training image file.
# output_model_path: path to save the weights of the classifier model when training completes.

$ python train_nav_gpu.py --feature_model_path fcn8s_mit_sceneparsing_benchmark_best_model.pkl --classifier_model_path last_classifier.pkl --train_csv_path train.csv --batch_size 6 --num_epoch 50 --output_model_path classifier.pkl
```

### Evaluate Classifier Model

```
# An example of test_less.csv
#  image,label
#  1539288123.627410.jpg,0
#  1539288340.287551.jpg,0
#  1539288516.918246.jpg,0
#  1539288178.703583.jpg,0
#  1539288314.283600.jpg,0
#  1539288439.401871.jpg,0
#  1539288561.424041.jpg,0
#  1539288121.627382.jpg,0
#  1539288795.435123.jpg,0
#  1539288427.875464.jpg,0
#  1539288306.773556.jpg,1
#  1539288332.772561.jpg,1
#  1539288303.275550.jpg,1
#  1539288143.655553.jpg,1
#  1539288145.639663.jpg,1
#  1539288739.931212.jpg,1
#  1539288146.659526.jpg,1
#  1539288549.963541.jpg,1
#  1539288324.766511.jpg,1
#  1539288300.767501.jpg,1
#  1539288466.889915.jpg,2
#  1539288088.609339.jpg,2
#  1539288722.449495.jpg,2
#  1539288814.945505.jpg,2
#  1539288668.929325.jpg,2
#  1539288762.930498.jpg,2
#  1539288849.428862.jpg,2
#  1539288060.579959.jpg,2
#  1539288510.418192.jpg,2
#  1539288697.430087.jpg,2

python train_nav.py --feature_model_path fcn8s_mit_sceneparsing_benchmark_best_model.pkl --classifier_model_path classifier.pkl --test_csv_path test_less.csv
```

Outputs accuracy (26.67% in this case):

```
Read testing csv file from : test_less.csv
accuracy tensor(0.2667, dtype=torch.float64)
```
