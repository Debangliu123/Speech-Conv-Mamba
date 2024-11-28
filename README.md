# Speech Conv-Mamba
Selective Structured State Space Model with Temporal Dilated Convolution for Efficient Speech Separation

### 1.Install Asteroid project

-  Stage1: get Asteroid project files 
(This process refers to the asteroid project: https://github.com/asteroid-team/asteroid)
```bash
git clone https://github.com/asteroid-team/asteroid
```

-  Stage2: pip asteroid to install the dependent environment
```bash
pip install asteroid
```

-  Stage3: Find  the model under the librimix dataset
You need to first go into the folder containing the asteroid project, then:
```bash
cd asteroid-master/egs/librimix
```
Now, you can find Comparison Models：

ConvTasNet

DPRNNTasNet

DPTNet

SuDORMRFNet

### 2.Install  Speech Conv-Mamba project

-  Stage1: get  Speech Conv-Mamba project files
```bash
git clone https://github.com/Debangliu123/Speech-Conv-Mamba
```
-  Stage2: Find the model 
```bash
cd audio_only_UBImamba_prj/src
```
Now, you can find：

AFRCNN (This model comes from the original paper project link: https://github.com/JusperLee/AFRCNN-For-Speech-Separation)

Sepformer(Sepformer_Wrapper.py)
(This model comes from the speechbrain project: https://github.com/speechbrain/speechbrain/)

Speech Conv-Mamba (*Our model will be uploaded after the final review)

### 3.Training
-  1.Training pipeline in asteroid
For example, training SuDORMRFNet:
```bash
cd asteroid-master/egs/librimix/SuDORMRFNet/run.sh
```
According to the run.sh file provided by Asteroid, we can easily train these models on the libri2Mix dataset (Please make sure you have downloaded the librimix dataset). 

-  2.Training pipeline in Speech Conv-Mamba (https://github.com/kaituoxu/Conv-TasNet)


Stage1：
```bash
cd   audio_only_UBImamba_prj/src
```

Stage2：
Run Audio_only_train1.py to perform training. (The training process partly refers to the project: https://github.com/kaituoxu/Conv-TasNet)

*Please note that in the librimix dataset (Please make sure you have downloaded the librimix dataset and generated a training set according to the instructions of the Asteroid project), using：
```
from  AVdata_LoadLandmark_for_librimix  import AudioandVideoDataLoader, AudioandVideoDataset
from Audio_only_solver_for_audio_only import  Solver
```

*and in the GRID dataset, using： 
```
from Audio_visual_solver_for_audio_only import Solver
from AVdata_LoadLandmark1 import AudioandVideoDataLoader, AudioandVideoDataset
```




