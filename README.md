
<div align="center">
<h1>ADAPT: Anchored Multimodal Physiological Transformer</h1>
<a href="https://www.python.org/"><img alt="PyTorch" src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra 1.3" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
</div>

Implementation of ADAPT @ MIDL 2024  [[Paper]](https://openreview.net/pdf?id=WDZg4P97gr) [[Project Page]](https://jumdc.github.io/adapt/)


<img width="1049" alt="Screenshot 2024-02-07 at 11 02 58" src="https://github.com/jumdc/ADAPT/assets/62952163/15fb6500-94b5-4237-94d5-0670a1b4b8d7">


# Project Structure
```
├── configs                      Hydra configs
│   ├── machine                  Machine configs (gpu, ...)
│   ├── model                    Model cfg
│   ├── multimodal               Data configs
│   ├── paths                    Project paths configs
│   ├── config.yaml              Main config for training
│   └── video-extract.yaml       Config for video feature extraction 
│
├── src                    
│   ├── datamodule                Data
│   │   ├── datasets             
│   │   └── multimodal_datamodule.py        
│   │
│   ├── models   
|   |   ├── modules               Modules used in the model
|   |   └── adapt.py              ADAPT model       
│   │     
│   └── utils   
│       ├── evaluation          
|       ├── preprocessing     
|       └── training                  
│
├── .gitignore                   List of files ignored by git
├── requirements.txt             File for installing python dependencies
├── train.py                     Main script for training
├── License                      
└── README.md
```

# 🚀 Quickstart
## Set-Up the environment
- Install Anaconda or MiniConda
- Run `conda create -n multi python=3.9`
- Activate multi: `conda activate multi`
- Install pytorch 1.12 and torchvision 0.13 that match your device
    - For GPU: 
    `conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge`
- Dependencies apart from `pytorch` may be install with the `pip isntall -r requirements.txt`.

## Prepare the data
- For the dataset, $\texttt{StressID}$ information and data request can be found [here](https://project.inria.fr/stressid/).
- Put the data in a directory called "StressID"
- Change the path in the cfg/paths/directories.yaml
- Extract the features from the videos. 


1. Training video features (using a sliding window)
```bash
python src/utils/preprocessing/video_features.py name=StressID_Dataset/train_video_features.txt
```
2. Testing video features (no sliding window)
```bash
python src/utils/preprocessing/video_features.py name=StressID_Dataset/test_video_features.txt  dataset.video.window=null dataset.hyperparams.batch_size=1 dataset.video.step=null
```

## Train ADAPT

The code is adapted to wandb logger, if you wish to use a logger make sure to be logged in to wandb before starting.
To run our code: 

```bash
python train.py
```

To disable the logger:
```bash
python train.py log=False
```

# Bibtex
If you happen to find this code or method useful in your research please cite this paper with: 
```latex
@inproceedings{
    mordacq2024adapt,
    title={{ADAPT}: Multimodal Learning for Detecting Physiological Changes under Missing Modalities},
    author={Julie Mordacq and Leo Milecki and Maria Vakalopoulou and Steve Oudot and Vicky Kalogeiton},
    booktitle={Submitted to Medical Imaging with Deep Learning},
    year={2024},
    url={https://openreview.net/forum?id=WDZg4P97gr},
    note={under review}
}
```