# <center>COVID-DA: Deep Domain Adaptation from Typical Pneumonia to COVID-19</center>
This repository provides the official implementation for "**COVID-DA: Deep Domain Adaptation from Typical Pneumonia to COVID-19**".

# Paper
[COVID-DA: Deep Domain Adaptation from Typical Pneumonia to COVID-19](https://arxiv.org/abs/2005.01577).

# Getting Started
## Installation
- Clone this repository:
```
git clone https://github.com/qiuzhen8484/COVID-DA.git
cd COVID-DA
```

- Install the requirements by runing the following command:
```
pip install -r requirements.txt
```

## Data Preparation
- The `.pkl` files of data list and its corresponding labels have been put in the directory `./data`.

- The other datasets, i.e., COVID-19-CT-scans and AMD, is available [here](https://github.com/qiuzhen8484/TDDA).

## Training
- To train COVID-DA, run the following command:
```
python main --gpu 0,1 --batchsize 16 --epoch 200 --path_source all_data_pneumonia --path_target all_data_covid
```

## Testing 
To test COVID-DA on the COVID-19 dataset using a well-trained model (please modify the argument "model_path"), run the following command:
```
python test_only --gpu 0,1 --batchsize 16 --model_path ./model_checkpoint/trained_best_model.pkl --path_target all_data_covid
```

# Citation
If you find our work useful in your research, please cite the following paper:
```
@article{zhang2020covidda,
    title={COVID-DA: Deep Domain Adaptation from Typical Pneumonia to COVID-19},
    author={Yifan Zhang and Shuaicheng Niu and Zhen Qiu and Ying Wei and Peilin Zhao and Jianhua Yao and Junzhou Huang and Qingyao Wu and Mingkui Tan},
    journal={arXiv},
    year={2020},
}
```
