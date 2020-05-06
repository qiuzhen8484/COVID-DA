# <center>COVID-DA: Deep Domain Adaptation from Typical Pneumonia to COVID-19</center>
We provide the dataset for domain adaptation from typical pneumonia to COVID-19. The paper is available [here](https://arxiv.org/pdf/2005.01577.pdf).

# Dataset
The descriptions for the dataset are presented below. We first provide a link to download the dataset. Next, the data statistics and usage of the dataset will be introduced.
## Download
- The dataset in this paper is available [here](http://suo.im/6d3jZF).

## Data structure and statistics
- The data structure:
```
all_data
└── all_data_pneumonia
|   |
|   ├── train
|   └── test 
|
└── all_data_covid
    |
    |── train
    |── val
    └── test
```

- Statistics of the dataset are shown as follow:

![data statistic](data.png "statistics of the dataset")

Pneumonia ("all_data_pneumonia" sub-directory) serves as the source domain and COVID-19 ("all_data_covid" sub-directory) serves as the target domain.

- You can refer to the [paper](https://arxiv.org/pdf/2005.01577.pdf) for more details about the dataset.

## Usage
- In the directory `./data`, there are two `.pkl` files which record the image lists and its corresponding labels. Specifically, an image and its label is stored in a tuple (image_name, label). "1" denotes class "pneumonia" and class "COVID-19" in source and target domain, respectively, while "0" denotes class "normal". You can read the data list following the below manner:
  - for the source domain Pneumonia:
  ```
        with open('./data/pneumonia_task.pkl', 'rb') as f:
            train_dict = pkl.load(f)
        train_list = train_dict['train_list'] # train sub-directory
        val_list = train_dict['val_list'] # test sub-directory
  ```
  - for the target domain COVID-19:
  ```
        with open('./data/COVID-19_task.pkl', 'rb') as f:
            train_dict = pkl.load(f)
        train_list_label = train_dict['train_list_semi'] # labeled data (train sub-directory)
        train_list_unlabel = train_dict['train_list'] # unlabeled data (train sub-directory)
        val_list = train_dict['val_list'] # val sub-directory
        test_list = train_dict['test_list'] # test sub-directory
  ```

# Citation
If you find the dataset useful, please cite the following paper:
```
@article{zhang2020covidda,
    title={COVID-DA: Deep Domain Adaptation from Typical Pneumonia to COVID-19},
    author={Yifan Zhang and Shuaicheng Niu and Zhen Qiu and Ying Wei and Peilin Zhao and Jianhua Yao and Junzhou Huang and Qingyao Wu and Mingkui Tan},
    journal={arXiv},
    year={2020},
}
```