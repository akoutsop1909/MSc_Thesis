# Monocular depth estimation using deep neural models
This repository contains the tweaked code of BANet, LapDepth, and PixelFormer to train on DIODE/Outdoor. To run the models, it is recommended to install Conda. Instructions [here](https://docs.anaconda.com/free/anaconda/). Alternatively, the link in the "about" section opens a google colab notebook (with links to other notebooks) to view datasets preparation and training results as well as a brief description (link will be provided soon).

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
pip install tensorboard matplotlib progressbar2 pandas opencv-python==4.5.1.48 opencv-contrib-python==3.4.4.19
```
> [!TIP]
> To remove the environment, enter the following command.
> ```
> conda env remove -n banet
> ```

### Training
* Train BANet on DIODE/Outdoor
  1. Open ```datasets_banet_csvs.ipynb``` from the ```scripts``` folder.
  2. Modify the path to the ```DIODE``` folder in the **Define functions** code cell.
  3. Run **Import packages**, **Define functions**, and **Create DIODE/Outdoor CSV files**.
  4. Replace ```diode_train.csv``` and ```diode_val.csv``` in ```models/BANet/datasets/``` with the new CSV files.
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
  2. Run **Import packages**, **Set Current Working Directory**, **Copy raw images to new location**, **Copy and convert depth PNG files to NPY**, and **Create KITTI Selection CSV file (relative path)**.
  3. Open ```datasets_banet_csvs.ipynb``` from the ```scripts``` folder.
  4. Modify the path to the ```kitti_selection``` folder in the **Create KITTI Selection CSV file** code cell.
  5. Run **Import packages** and **Create KITTI Selection CSV file**.
  6. Replace ```kitti_selection_banet.csv``` in ```models/BANet/datasets/``` with the new CSV file.
  7. You can now execute the evaluation command (modify [path_to_model]).
```
python3 main.py --inference random --height 192 --width 256 --model [path_to_model] --val_csv datasets/kitti_selection_banet.csv
```
> [!TIP]
> Steps i. and ii. are the same for all three models, so you do not need to execute them again if you have already done so.

* Evaluate IHU
  1. Open ```datasets_banet_csvs.ipynb``` from the ```scripts``` folder.
  2. Modify the path to the ```ihu_resized``` folder in the **Create IHU CSV file** code cell.
  3. Run **Import packages** and **Create IHU CSV file**.
  4. Replace ```ihu_banet.csv``` in ```models/BANet/datasets/``` with the new CSV file.
  5. You can now execute the evaluation command (modify [path_to_model]).
```
python3 main.py --inference random --height 192 --width 256 --model [path_to_model] --val_csv datasets/ihu_banet.csv
```

## Training/Evaluation with LapDepth
### Installation
```
conda create -n lapdepth python=3.7 anaconda
conda activate lapdepth
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=9.2 -c pytorch
pip install geffnet path IPython blessings progressbar
```
> [!TIP]
> To remove the environment, enter the following command.
> ```
> conda env remove -n lapdepth
> ```
### Training
* Train LapDepth on DIODE/Outdoor
  1. Place ```train_diodeaskitti.txt``` and ```val_diodeaskitti.txt``` from ```DIODEASKITTI``` into the ```DIODE``` folder.
  2. Open ```datasets_diodeaskitti.ipynb``` from the ```scripts``` folder and run **Import packages**, **Define functions**, **Create DIODEASKITTI raw filder structure**, and **Create DIODEASKITTI depth folder structure**.
  3. Use the new dataset ```DIODEASKITTI```.
  4. You can now execute the training command (modify [path_to_DIODEASKITTI]).
```
python train.py --distributed --val_in_train --epochs 45 --max_depth 300.0 --weight_decay 1e-1 --dataset KITTI --data_path [path_to_DIODEASKITTI] --gpu_num 0,1,2,3
```
> [!TIP]
> Steps i. and ii. are the same for LapDepth and PixelFormer, so you do not need to execute them again if you have already done so.

### Evaluation
* Evaluate DIODE/Outdoor
  1. Execute the evaluation command (modify [path_to_model] and [path_to_DIODEASKITTI]).
```
python eval.py --model_dir [path_to_model] --img_save --evaluate --batch_size 1 --dataset KITTI --data_path [path_to_DIODEASKITTI] --gpu_num 0
```

* Evaluate KITTI Selection
  1. Open ```datasets_kitti_selection.ipynb``` from the ```scripts``` folder.
  2. Run **Import packages**, **Set Current Working Directory**, **Copy raw images to new location**, **Copy and convert depth PNG files to NPY**, and **Create KITTI Selection CSV file (relative path)**.
  3. Place the raw, RGB images into ```DIODEASKITTI/2011_09_26/2011_09_26_drive_0000_sync/image_02/data/```. You will need to create the folder structure manually.
  4. Place the PNG depth maps into ```DIODEASKITTI/data_depth_annotated/2011_09_26_drive_0000_sync/proj_depth/groundtruth/image_02/```. You will need to create the folder structure manually.
  5. You can now execute the evaluation command (modify [path_to_model] and [path_to_DIODEASKITTI]).
```
python eval.py --model_dir [path_to_model] --img_save --evaluate --batch_size 1 --dataset KITTI --testfile_kitti ./datasets/kitti_selection.txt --data_path [path_to_DIODEASKITTI] --gpu_num 0
```
> [!TIP]
> Steps i. and ii. are the same for all three models, so you do not need to execute them again if you have already done so.<br>
> Step iii. is also the same for LapDepth and PixelFormer.

* Evaluate IHU
  1. Place the RGB images into ```DIODEASKITTI/2011_09_26/2011_09_26_drive_ihu_sync/image_02/data/```. You will need to create the folder structure manually.
  2. Place the fake PNG depth map into ```DIODEASKITTI/data_depth_annotated/2011_09_26_drive_ihu_sync/proj_depth/groundtruth/image_02/```. You will need to create the folder structure manually.
  3. You can now execute the evaluation command (modify [path_to_model] and [path_to_DIODEASKITTI]).
```
python eval.py --model_dir [path_to_model] --img_save --evaluate --batch_size 1 --dataset KITTI --testfile_kitti ./datasets/ihu.txt --data_path [path_to_DIODEASKITTI] --gpu_num 0
```

## Training/Evaluation with PixelFormer
### Installation
```
conda create -n pixelformer python=3.8
conda activate pixelformer
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
pip install matplotlib tqdm tensorboardX timm mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
```
> [!TIP]
> To remove the environment, enter the following command.
> ```
> conda env remove -n pixelformer
> ```
### Training
* Train PixelFormer on DIODE/Outdoor
  1. Download ```swin_large_patch4_window7_224_22k.pth``` from [here](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth) and place it into ```models/PixelFormer/pretrained/```. 
  2. Place ```train_diodeaskitti.txt``` and ```val_diodeaskitti.txt``` from ```DIODEASKITTI``` into the ```DIODE``` folder.
  3. Open ```datasets_diodeaskitti.ipynb``` from the ```scripts``` folder and run **Import packages**, **Define functions**, **Create DIODEASKITTI raw folder structure**, and **Create DIODEASKITTI depth folder structure**.
  4. Create a folder ```2011_09_26``` in ```DIODEASKITTI/data_depth_annotated/``` and place the ```2011_09_26_drive_xxxx_sync``` folders from ```data_depth_annotated``` into the new folder.
  5. Open ```arguments_train_kittieigen.txt``` from ```models/PixelFormer/configs/``` and modify the path to the ```DIODEASKITTI``` folder of the **data_path** and **data_path_eval** hyperparameters. Also modify the path to ```DIODEASKITTI/data_depth_annotated/``` of the **gt_path** and **gt_path_eval** hyperparameters.
  6. Use the new dataset ```DIODEASKITTI```.
  7. You can now execute the training command.
```
python pixelformer/train.py configs/arguments_train_kittieigen.txt
```
> [!TIP]
> Steps ii. and iii. are the same for LapDepth and PixelFormer, so you do not need to execute them again if you have already done so.

### Evaluation
* Evaluate DIODE/Outdoor
  1. Open ```arguments_eval_kittieigen.txt``` from ```models/PixelFormer/configs/``` and modify the path to the ```DIODEASKITTI``` folder of **data_path_eval**, the path to ```DIODEASKITTI/data_depth_annotated/``` of **gt_path_eval**, and the path to the model of **checkpoint_path**.
  2. You can now execute the evaluation command.
```
python pixelformer/eval.py configs/arguments_eval_kittieigen.txt
```

* Evaluate KITTI Selection
  1. Open ```datasets_kitti_selection.ipynb``` from the ```scripts``` folder.
  2. Run **Import packages**, **Set Current Working Directory**, **Copy raw images to new location**, **Copy and convert depth PNG files to NPY**, and **Create KITTI Selection CSV file (relative path)**.
  3. Place the raw, RGB images into ```DIODEASKITTI/2011_09_26/2011_09_26_drive_0000_sync/image_02/data/```. You will need to create the folder structure manually.
  4. Place the PNG depth maps into ```DIODEASKITTI/data_depth_annotated/2011_09_26/2011_09_26_drive_0000_sync/proj_depth/groundtruth/image_02/```. You will need to create the folder structure manually.
  5. Open ```arguments_eval_kittieigen.txt``` from ```models/PixelFormer/configs/``` and modify the path to the ```DIODEASKITTI``` folder of **data_path_eval**, the path to ```DIODEASKITTI/data_depth_annotated/``` of **gt_path_eval**, and the path to the model of **checkpoint_path**. Also set **filenames_file_eval** to ```data_splits/kitti_selection.txt```.
  6. You can now execute the evaluation command.
```
python pixelformer/eval.py configs/arguments_eval_kittieigen.txt
```
> [!TIP]
> Steps i. and ii. are the same for all three models, so you do not need to execute them again if you have already done so.<br>
> Step iii. is also the same for LapDepth and PixelFormer.

* Evaluate IHU
  1. Place the RGB images into ```DIODEASKITTI/2011_09_26/2011_09_26_drive_ihu_sync/image_02/data/```. You will need to create the folder structure manually.
  2. Place the fake PNG depth map into ```DIODEASKITTI/data_depth_annotated/2011_09_26/2011_09_26_drive_ihu_sync/proj_depth/groundtruth/image_02/```. You will need to create the folder structure manually.
  3. Open ```arguments_eval_kittieigen.txt``` from ```models/PixelFormer/configs/``` and modify the path to the ```DIODEASKITTI``` folder of **data_path_eval**, the path to ```DIODEASKITTI/data_depth_annotated/``` of **gt_path_eval**, and the path to the model of **checkpoint_path**. Also set **filenames_file_eval** to ```data_splits/ihu.txt```.
  4. You can now execute the evaluation command.
```
python pixelformer/eval.py configs/arguments_eval_kittieigen.txt
```
> [!TIP]
> Step i. is the same for LapDepth and PixelFormer, so you do not need to execute it again if you have already done so.
