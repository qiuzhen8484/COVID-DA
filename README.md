# <center>COVID-DA: Deep Domain Adaptation from Typical Pneumonia to COVID-19</center>
We will provide PyTorch implementation for "**COVID-DA: Deep Domain Adaptation from Typical Pneumonia to COVID-19**".

# Paper
The paper will be provided.

# Dataset
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
Pneumonia serves as the source domain and COVID-19 serves as the target domain.

- You can refer to the paper for more details about the dataset.

## Usage
- In the directory `./data`, there are two `.pkl` files which record the image lists and its corresponding labels. Specifically, an image and its label is stored in a tuple (image_name, label). "1" denotes class pneumonia and class COVID-19 in source and target domain, respectively, while "0" denotes normal. You can read the data list following the below manner:
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
