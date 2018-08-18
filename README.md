# PIRM2018-SpectralSR-Team-Grit
The repo now contains the testing codes which enable user to validate/test the pretrained model on testing/validation set

## Installation
The main dependencies are listed as follows (others are left but should also be installed if missed)
```
conda install pytorch=0.4.0  
pip install torchnet torchvision
```

The pretrained models could be downloaded from [this path](https://drive.google.com/drive/folders/1Il7mXkuJMr77Bs-NoFjiF2K46i7ukiFk?usp=sharing),
which contains the EMSR and EMSR-CA models we submitted in final testing stage of PIRM2018-Spectral-SR challenge.

## Testing
The procedures of testing our pretrained models are listed as follows:

1. use **envi2mat.m** to transform your envi data into mat data  

2. modify the pathes in **hsi_test.py**

3. finally, run the code below
```
python hsi_test.py -a emsrx3 -r -rp /path/to/emsrx3/model_best.pth --sf 3 --self-ensemble --test --no-log -nro
python hsi_test.py -a emsrcax3 -r -rp /path/to/emsrcax3/model_best.pth --sf 3 --self-ensemble --test --no-log -nro
```

If you have any questions, please do not hesitate to contact me (kaixuan_wei@outlook.com).