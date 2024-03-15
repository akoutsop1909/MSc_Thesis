# Monocular depth estimation using deep neural models
This repository contains the tweaked code of BANet, LapDepth, and PixelFormer to train on DIODE/Outdoor. To run the models, it is recommended to install Conda. Instructions are provided [here](https://docs.anaconda.com/free/anaconda/). Alternatively, the link in the "about" section opens a google colab notebook (with links to other notebooks) to view training statistics and predicted depth maps as well as a brief description (link will be provided soon).

The original GitHub repositories can be found [here](https://github.com/dg-enlens/banet-depth-prediction) (BANet), [here](https://github.com/tjqansthd/LapDepth-release) (LapDepth), and [here](https://github.com/ashutosh1807/PixelFormer) (PixelFormer).

In summary, the MSc thesis compared BANet, LapDepth, and PixelFormer: three state-of-the-art deep learning models that estimate depth from a single image. The models were trained from scratch on DIODE/Outdoor with adjustments to the code to load the dataset. Different hyperparameters were tested  to find the combination that yields the best statistics for each model. The SILog loss and RMSE criteria ensured an objective evaluation. In addition, two more datasets (KITTI Selection and IHU) provided a subjective evaluation.

The PDF file of the MSc thesis ```Monocular depth estimation using deep neural models.pdf``` is also available.
## Datasets
### DIODE/Outdoor (training, validation, test)
DIODE/Outdoor is a subset of DIODE. It contains 16,884 training and 446 validation images and depth maps tottaling 124GB. Both were captured wth the FARO Focus S350 laser scanner with a range of 350 meters. DIODE/Outdoor's diverse scenes depict city streets, large residential buildings, hiking trails, parking lots, parks, forests, river banks, and other outdoor environments at short, medium, and long distances. 

The dataset can be downloaded from [here](https://diode-dataset.org).

### KITTI Selection (test)
KITTI Selection is the validation set of KITTI, mostly intended for autonomous driving applications. It contains 1,000 images and depth maps tottaling 802MB. A Volkswagen station wagon with mounted equipment drove around Karlsruhe, Germany, to record various driving scenarios. The setup included a Velodyne HDL-64E LiDAR sensor with a range of 120 meters and a PointGray Flea2 FL2-14S3C-C color camera.

The dataset can be downloaded from [here](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction).

### IHU (test)

## Training/Evaluation with BANet
### Dataset preparation

### Installation

### Training

### Evaluation

## Training/Evaluation with LapDepth
### Dataset preparation

### Installation

### Training

### Evaluation

## Training/Evaluation with PixelFormer
### Dataset preparation

### Installation

### Training

### Evaluation
