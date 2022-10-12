<meta name="robots" content="noindex">

# LightCTS: A Lightweight Framework for Correlated Time Series Forecasting

This is the repository of the paper entitled "LightCTS: A Lightweight Framework for Correlated Time Series Forecasting", encompassing the code, datasets, and supplemental material.

# Supplemental Material

Detailed time and space complexity analysis and other groups of experimental results can be found at the [Supplemental Material](Supplemental_Material/Supplemental_Material_to_the_paper_LightCTS.pdf) (download recommended for better readability).

  
  
  
  <br>
  <br>
  <br>
  <br>
  <br>
  <br>
  

# Code and Datasets

## Requirements

To install requirements:

```setup
pip3 install -r requirements.txt
```

## Multi-step Forecasting
### Datasets
LightCTS is implemented on four public multi-step correlated time series forecasting datasets.

- **PEMS04** and **PEMS08** from STSGCN (AAAI-20). Download the data [STSGCN_data.tar.gz](https://pan.baidu.com/s/1ZPIiOM__r1TRlmY4YGlolw) with password: `p72z` and uncompress data file using`tar -zxvf data.tar.gz`, and move them to the data folder.

- **METR-LA** and **PEMS-BAY** from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN), please follow its instructions for data pre-processing, and move them to the data folder.

- **Solar** and **Electricity** datasets from [https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data). Uncompress them and move them to the data folder.

### Baselines

| Model      | Conference | Year | Link                                                  |
|------------|------------|------|-------------------------------------------------------|
| DCRNN      | ICLR       | 2018 | https://openreview.net/pdf?id=SJiHXGWAZ               |
| GWNET      | IJCAI      | 2019 | https://dl.acm.org/doi/10.5555/3367243.3367303        |
| AGCRN      | NeurIPS    | 2020 | https://dl.acm.org/doi/abs/10.5555/3495724.3497218    |
| MTGNN      | KDD        | 2020 | https://dl.acm.org/doi/abs/10.1145/3394486.3403118    |
| AUTOCTS    | VLDB       | 2021 | https://dl.acm.org/doi/10.14778/3503585.3503604       |
| EnhanceNet | ICDE       | 2021 | https://ieeexplore.ieee.org/abstract/document/9458855 |
| FOGS       | IJCAI      | 2022 | https://www.ijcai.org/proceedings/2022/545            |

### Model Training and Testing
* PEMS04/PEMS08
```
cd Multi-step/Traffic Flow/{dataset_name}/
python train_{dataset_name_in_lowercase}.py
python test_{dataset_name_in_lowercase}.py -checkpoint {path_to_the_checkpoint_file}
```
* METR-LA/PEMS-BAY
```
cd Multi-step/Traffic Speed/{dataset_name}/
python train_{dataset_name_in_lowercase}.py
python test_{dataset_name_in_lowercase}.py -checkpoint {path_to_the_checkpoint_file}
```
## Single-step Forecasting
### Datasets
LightCTS is implemented on two public single-step correlated time series forecasting datasets.

- **Solar** and **Electricity** datasets from [https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data). Uncompress them and move them to the data folder.

### Baselines

| Model      | Conference | Year | Link                                                  |
|------------|------------|------|-------------------------------------------------------|
| DSANet     | CIKM       | 2019 | https://dl.acm.org/doi/abs/10.1145/3357384.3358132    |
| MTGNN      | KDD        | 2020 | https://dl.acm.org/doi/abs/10.1145/3394486.3403118    |
| AUTOCTS    | VLDB       | 2021 | https://dl.acm.org/doi/10.14778/3503585.3503604       |
| MAGNN      | arXiv      | 2022 | https://arxiv.org/abs/2201.04828                      |

### Model Training and Testing
* Solar-Energy/Electricity
```
cd Single-step/{dataset_name}/
python train_{dataset_name_in_lowercase}.py
python test_{dataset_name_in_lowercase}.py -checkpoint {path_to_the_checkpoint_file}
```
## Pre-trained checkpoint files
Pre-trained checkpoint files can be download from [Google Drive](https://drive.google.com/drive/folders/1_-jAQciSdPiI8wKkfvIvlHRk1Rnx-LJC?usp=sharing). Please add the checkpoint file path to corresponding "args.checkpoint" of the test code file.



