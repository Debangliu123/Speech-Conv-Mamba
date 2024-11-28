# Speech Conv-Mamba
Selective Structured State Space Model with Temporal Dilated Convolution for Efficient Speech Separation



###  Stage1: get Asteroid project files
```bash
git clone https://github.com/asteroid-team/asteroid
```


###  Stage2: pip asteroid to install the dependent environment
```bash
pip install asteroid
```

###  Stage3: Find  the model under the librimix dataset
You need to first go into the folder containing the asteroid project, then:
```bash
cd asteroid-master/egs/librimix
```
Now, you can find Comparison Models：
- ConvTasNet
- DPRNNTasNet
- DPTNet
- SuDORMRFNet
and
```bash
cd audio_only_UBImamba_prj/src
```
Now, you can find：
- Sepformer(Sepformer_Wrapper.py)
- Speech Conv-Mamba (Our model will be uploaded after the final review)


###  Stage4: Training
For example, training SuDORMRFNet:
```bash
cd asteroid-master/egs/librimix/SuDORMRFNet/run.sh
```
According to the run.sh file provided by Asteroid, we can easily train these models on the libri2Mix dataset (Please make sure you have downloaded the librimix dataset). 



