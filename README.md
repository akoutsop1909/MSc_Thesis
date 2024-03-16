# Monocular depth estimation using deep neural models
This repository contains the tweaked code of BANet, LapDepth, and PixelFormer to train on DIODE/Outdoor. To run the models, it is recommended to install Conda. Instructions are provided [here](https://docs.anaconda.com/free/anaconda/). Alternatively, the link in the "about" section opens a google colab notebook (with links to other notebooks) to view datasets preparation and training results as well as a brief description (link will be provided soon).

The original GitHub repositories can be found [here](https://github.com/dg-enlens/banet-depth-prediction) (BANet), [here](https://github.com/tjqansthd/LapDepth-release) (LapDepth), and [here](https://github.com/ashutosh1807/PixelFormer) (PixelFormer).

In summary, the MSc thesis compared the above three state-of-the-art deep learning models that estimate depth from a single image. The models were trained from scratch on DIODE/Outdoor with adjustments to the code to load the dataset. Different hyperparameters were tested  to find the combination that yields the best statistics for each model. The SILog loss and RMSE criteria ensured an objective evaluation. In addition, two more datasets (KITTI Selection and IHU) provided a subjective evaluation.

The PDF file of the MSc thesis ```Monocular depth estimation using deep neural models.pdf``` is also available.
## Datasets
### DIODE/Outdoor (training, validation, test)
DIODE/Outdoor is a subset of DIODE. It contains 16,884 training and 446 validation images and depth maps tottaling 124GB. Both were captured wth the FARO Focus S350 laser scanner with a range of 350 meters. DIODE/Outdoor's diverse scenes depict city streets, large residential buildings, hiking trails, parking lots, parks, forests, river banks, and other outdoor environments at short, medium, and long distances. 

The dataset can be downloaded from [here](https://diode-dataset.org).

### KITTI Selection (test)
KITTI Selection is the validation set of KITTI, mostly intended for autonomous driving applications. It contains 1,000 images and depth maps tottaling 802MB. A Volkswagen station wagon with mounted equipment drove around Karlsruhe, Germany, to record various driving scenarios. The setup included a Velodyne HDL-64E LiDAR sensor with a range of 120 meters and a PointGray Flea2 FL2-14S3C-C color camera.

The dataset can be downloaded from [here](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction).

### IHU (test)
The IHU dataset was created specifically for this thesis. It contains 30 RGB images taken after a walk around the Alexandrian Campus of IHU on the 19th of October 2023 from 6 p.m. to 7:30 p.m. A Samsung Galaxy A13 5G smartphone was the only equipment. The aspect ratio was set to 4:3, resulting in 3060x4080 resolution RGB images. They were then resized to 1024x768 tottaling 4MB and DIODE’s resolution into consideration. Moreover, IHU depicts similar outdoor environments to DIODE/Outdoor with objects, such as vehicles, roads, trees, and signs.

The resized dataset can be found in the ```IHU``` folder of this repository.
## Training/Evaluation with BANet
### Installation
```
conda create -n banet python=3.6.8 anaconda
conda activate banet
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install opencv-python==4.5.1.48 opencv-contrib-python==3.4.4.19
```

### Training
* Train BANet on DIODE/Outdoor
  1. Open ```datasets_banet_csvs.ipynb``` from the ```scripts``` folder.
  2. Modify the path to the DIODE folder in the **Define functions** code cell.
  3. Run **Import packages**, **Define functions**, and **Create DIODE/Outdoor CSV files**.
  4. Replace ```diode_train.csv``` and ```diode_val.csv``` in ```models/BANet/datasets``` with the newly created csv files.
  5. You can now execute the training command.
```
python3 main.py --train 1 --height 192 --width 256 --train_csv datasets/diode_train.csv --val_csv datasets/diode_val.csv
```

### Evaluation
* Evaluate DIODE/Outdoor
  1. Execute the evaluation command (modify [path_to_model]).
```
python3 main.py --inference random --height 192 --width 256 --model [path_to_model] --val_csv datasets/diode_val.csv
```
* Evaluate KITTI Selection
  1. Open ```datasets_kitti_selection.ipynb``` from the ```scripts``` folder.
  2. Run **Import packages**, **Copy raw images to new location**, **Copy and convert depth PNG files to NPY**, and **Create KITTI Selection CSV file (relative path)**.
  3. Open ```datasets_banet_csvs.ipynb``` from the ```scripts``` folder.
  4. Modify the path to the kitti_selection folder in the **Create KITTI Selection CSV file** code cell.
  5. Run **Import packages** and **Create KITTI Selection CSV file**.
  6. Replace ```kitti_selection_banet.csv``` in ```models/BANet/datasets``` with the newly created csv file.
  7. You can now execute the evaluation command (modify [path_to_model]).
```
python3 main.py --inference random --height 192 --width 256 --model [path_to_model] --val_csv datasets/kitti_selection_banet.csv
```
* Evaluate IHU
  1. Open ```datasets_banet_csvs.ipynb``` from the ```scripts``` folder.
  2. Modify the path to the ihu_resized folder in the **Create IHU CSV file** code cell.
  3. Run **Import packages** and **Create IHU csv file**.
  4. Replace ```ihu_banet.csv``` in ```models/BANet/datasets``` with the newly created csv file.
  5. You can now execute the evaluation command (modify [path_to_model]).
```
python3 main.py --inference random --height 192 --width 256 --model [path_to_model] --val_csv datasets/ihu_banet.csv
```

## Training/Evaluation with LapDepth
### Installation

### Training

### Evaluation

## Training/Evaluation with PixelFormer
### Installation

### Training

### Evaluation
