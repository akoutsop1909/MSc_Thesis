# Monocular depth estimation using deep neural models
This repository contains the tweaked code of BANet, LapDepth, and PixelFormer to train on DIODE/Outdoor. To run the models, it is recommended to install Conda. Instructions are provided [here](https://docs.anaconda.com/free/anaconda/). Alternatively, the link in the "about" section opens a google colab notebook (with links to other notebooks) to view training statistics and predicted depth maps as well as a brief description (link will be provided soon).

In summary, the MSc thesis compared BANet, LapDepth, and PixelFormer: three state-of-the-art deep learning models that estimate depth from a single image. The models were trained from scratch on DIODE/Outdoor with adjustments to the code to load the dataset. Different hyperparameters were tested  to find the combination that yields the best statistics for each model. The SILog loss and RMSE criteria ensured an objective evaluation. In addition, two more datasets (KITTI Selection and IHU) provided a subjective evaluation.

The PDF file of the MSc thesis ```Monocular depth estimation using deep neural models.pdf``` is also available.
## Datasets
### DIODE/Outdoor (train, validation, test)
DIODE/Outdoor is a subset of DIODE. It contains 16,884 training and 446 validation images and depth maps with a total size of 124GB. Both were captured wth the FARO Focus S350 laser scanner with a range of 350 meters. DIODE/Outdoor's diverse scenes depict city streets, large residential buildings, hiking trails, parking lots, parks, forests, river banks, and other outdoor environments at short, medium, and long distances. 

The dataset can be downloaded from [here](https://diode-dataset.org)

### KITTI Selection (test)

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
