# Long Document Classification usin Model Fusing

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

## Benchmarking Results
You can easily run benchmarking using following command and altering the parameters as you wish

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

```

[//]: # (## Citation)

[//]: # (Please consider citing us if you use the library or the code. )

[//]: # (```bash)

[//]: # (@inproceedings{damith2022DTWquranqa,)

[//]: # (  title={DTW at Qur'an QA 2022: Utilising Transfer Learning with Transformers for Question Answering in a Low-resource Domain},)

[//]: # (  author={Damith Premasiri and Tharindu Ranasinghe and Wajdi Zaghouani and Ruslan Mitkov},)

[//]: # (  booktitle={Proceedings of the 5th Workshop on Open-Source Arabic Corpora and Processing Tools &#40;OSACT5&#41;.},)

[//]: # (  year={2022})

[//]: # (})

[//]: # (```)
