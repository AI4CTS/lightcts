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

- **PEMS04**, **PEMS08**, **METR-LA**, and **PEMS-BAY** can be downloaded in [Google Drive](https://drive.google.com/drive/folders/1EOuTv2w-9gaGGPm3CylMXTJ4mg3et7x7?usp=sharing). Uncompress and move them to the corresponding data folder.

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

- **Solar** and **Electricity** datasets can be downloaded in [Google Drive](https://drive.google.com/drive/folders/1JYwtq120bkI-ze85hvtqEc8EkBoGA_U2?usp=sharing). Uncompress and move them to the corresponding data folder.


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

Pre-trained checkpoint files can be download from [Google Drive](https://drive.google.com/drive/folders/1rUfYkcWKsbXqSSFgn2J6wOctkEx2qmzk?usp=sharing). Please replace "args.checkpoint" with the corresponding path in the test code file.

