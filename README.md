# Long Document Classification using Model Fusing

An emperical study on utilising model fusing for long document classification.

## Installation
First you need to install PyTorch. The recommended PyTorch version is 1.13.1
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) for more details specifically for the platforms.

When PyTorch has been installed, you can install requirements from source by cloning the repository and running:

```bash
git clone https://github.com/DamithDR/legal-classification.git
cd legal-classification
pip install -r requirements.txt
```

## Experiment Results
You can easily run experiments using following command and altering the parameters as you wish

```bash
python -m experiments.classification.classification_variations --n_fold 1 --device_number 0 --no_of_models 3 --dataset ECHR
```

## Baseline Results
You can easily run baselines using following command and altering the parameters as you wish

```bash
python -m experiments.classification.benchmarks --device_number 0 --no_of_models 3 --dataset ECHR
```

## Parameters
Please find the detailed descriptions of the parameters
```text
n_fold              : Number of executions expected before self ensemble
device_number       : Cuda device number; in case of multiple GPUs are visible
no_of_models        : No of sub-models used to fusing
dataset             : Alias of the dataset need to experiment
base_model          : Name of the base model which is used for submodels and fused model
model_type          : Type of the model; Ex: BERT
epochs              : Number of training epochs

```

## Supported Datasets
```text
ECHR                : European Court of Human Rights (ECHR) cases
ECHR_Anon           : Anonymize version of ECHR dataset having named entities anonymised
20_news_categories  : Dataset of 20 different news categories
case-2021           : Dataset from the shared Task on Socio-political and Crisis Events Detection CASE - subtask 1.

Note : Case-2021 dataset is not provided with this repository due to their restrictions of use of the data. 
Please contact the team on : https://github.com/emerging-welfare/case-2022-multilingual-event and place the english
dataset at data/processed/case-2021/ folder and make the data a Json list so that it can directly read using pandas.
This way you can use the case-2021 dataset as well. 

```

## Citation
Please consider citing us if you use the code or paper. 
```bash
@inproceedings{damith2023fusinglongdoc,
  title={Can Model Fusing Help Transformers in Long Document Classification? An Empirical Study.},
  author={Damith Premasiri and Tharindu Ranasinghe and Ruslan Mitkov},
  booktitle={Proceedings of the 14th Conference Recent Advances In Natural Language Processing (RANLP)},
  year={2023}
}
```
