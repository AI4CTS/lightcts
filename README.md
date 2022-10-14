<meta name="robots" content="noindex">

<h1> LightCTS: A Lightweight Framework for Correlated Time Series Forecasting </h1>

This is the repository of the paper entitled "LightCTS: A Lightweight Framework for Correlated Time Series Forecasting", encompassing the code, datasets, and supplemental material.

<h1> Supplemental Material </h1> 

Detailed time and space complexity analysis and other groups of experimental results can be found at the [Supplemental Material](Supplemental_Material/Supplemental_Material_to_the_paper_LightCTS.pdf) (downloading to local pdf viewer is recommended for better readability).

 <br>
 <br>

  

<h1> Code and Datasets </h1> 

<h2> Requirements </h2> 

To install requirements:

```setup
pip3 install -r requirements.txt
```

<h2> Multi-step Forecasting </h2>

<h3> Datasets </h3> 

LightCTS is implemented on four public multi-step correlated time series forecasting datasets.

- **PEMS04** and **PEMS08** from STSGCN (AAAI-20). Download the data [STSGCN_data.tar.gz](https://pan.baidu.com/s/1ZPIiOM__r1TRlmY4YGlolw) with password: `p72z` and uncompress data file using`tar -zxvf data.tar.gz`, and move them to the data folder.

- **METR-LA** and **PEMS-BAY** from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) originally provided by [DCRNN](https://openreview.net/pdf?id=SJiHXGWAZ), please follow its instructions in its code repository for data pre-processing, and move them to the data folder.

<h3> Baselines </h3> 

| Model      | Conference | Year | Link                                                  |
|------------|------------|------|-------------------------------------------------------|
| DCRNN      | ICLR       | 2018 | https://openreview.net/pdf?id=SJiHXGWAZ               |
| GWNET      | IJCAI      | 2019 | https://dl.acm.org/doi/10.5555/3367243.3367303        |
| AGCRN      | NeurIPS    | 2020 | https://dl.acm.org/doi/abs/10.5555/3495724.3497218    |
| MTGNN      | KDD        | 2020 | https://dl.acm.org/doi/abs/10.1145/3394486.3403118    |
| AUTOCTS    | VLDB       | 2021 | https://dl.acm.org/doi/10.14778/3503585.3503604       |
| EnhanceNet | ICDE       | 2021 | https://ieeexplore.ieee.org/abstract/document/9458855 |
| FOGS       | IJCAI      | 2022 | https://www.ijcai.org/proceedings/2022/545            |

<h3> Model Training and Testing </h3>

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

<h2> Single-step Forecasting </h2>

<h3> Datasets </h3>  

LightCTS is implemented on two public single-step correlated time series forecasting datasets.

- **Solar** and **Electricity** datasets from [Google Drive](https://drive.google.com/drive/folders/1sw0m6MOfglsCQKzP_NuLpELjlca9avBh?usp=sharing) originally provided by [LSTNet](https://arxiv.org/abs/1703.07015). Uncompress them and move them to the data folder.

<h3> Baselines </h3> 


| Model      | Conference | Year | Link                                                  |
|------------|------------|------|-------------------------------------------------------|
| DSANet     | CIKM       | 2019 | https://dl.acm.org/doi/abs/10.1145/3357384.3358132    |
| MTGNN      | KDD        | 2020 | https://dl.acm.org/doi/abs/10.1145/3394486.3403118    |
| AUTOCTS    | VLDB       | 2021 | https://dl.acm.org/doi/10.14778/3503585.3503604       |
| MAGNN      | arXiv      | 2022 | https://arxiv.org/abs/2201.04828                      |

<h3> Model Training and Testing </h3>

* Solar-Energy/Electricity
```
cd Single-step/{dataset_name}/
python train_{dataset_name_in_lowercase}.py
python test_{dataset_name_in_lowercase}.py -checkpoint {path_to_the_checkpoint_file}
```

<h2> Pre-trained checkpoint files </h2>

Pre-trained checkpoint files can be download from [Google Drive](https://drive.google.com/drive/folders/1_-jAQciSdPiI8wKkfvIvlHRk1Rnx-LJC?usp=sharing). Please replace "args.checkpoint" with the corresponding path in the test code file.

