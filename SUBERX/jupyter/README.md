# Jupyter Notebooks

This folder contains many jupyter notebooks to work with. It is mounted in from the source folder so any changes made will be reflected in the data.

# Notebooks

Here's a brief description of each notebook and what it does. 

## BehaviorAnalysis

The [BehaviorsAnalysis Jupyter Notebook](BehaviorAnalysis.ipynb) is a quick look into the destribution of data. More can be done here.  The [entity_mapping.csv](entity_mapping.csv) file is created and used by this notebook.

## System Prep

The [System_Prep Jupyter Notebook](System_Prep.ipynb) runs through some basic examples illustrating the frameworks being used in this worok.  This include **PyTorch**, **TensorFlow**, and **Stable Baselines 3**.  It was a challenge to get all three working in one container.  **Pydantic AI** Should be added to this.

## Fixing_data.ipynb

The [Fixing_data Jupyter Notebook](Fixing_data.ipynb) is here as a scrap-book to work with data if you need to. This notebook can be safely ignored or deleted.

## Synthetic Analysts 

The [Synthetic Analysts Jupyter Notebook](SyntheticAnalysts.ipynb) Generates the synthetic analysts. the source code is in the `synthetic_analyst.py` located in the base of the project -- `SUBERX/environment/LLM`.

## Synthetic Data 

The [Synthetic Data Jupyter Notebook](SyntheticData.ipynb) Generates the `behaviors` of the synthetic analysts.  

## The NRMS Notebooks

A notebook each was produced for demo, small, and large. These notebooks were slightly modified to account for the lack for funcitons that retrieved the datasets.  These at one point stopped working so it was necessary to retrieve the data and make it all work again without the fancy data retrieval functions.

**NRMS--"Neural News Recommendation with Multi-Head Self-Attention" Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie (EMNLP 2019).**


### NRMS Demo Dataset

The [NRMS Demo Dataset Jupyter Notebook](Nrms_demo_dataset.ipynb) applys the NRMS algorithm to a small sample of the data.  It illustrates the algorithm working. Time to run on my setup: 10 minutes.

### NRMS Small Dataset

The [NRMS Small Dataset Jupyter Notebook](Nrms_small_dataset.ipynb) applys the NRMS algorithm to a small subset of the large dataset. This is much more than the demo dataset. It illustates positive results for the algorithm.  Time to run on my setup: ~ 1 hour and 15 minutes.

### NRMS Large Dataset

The [NRMS Large Dataset Jupyter Notebook](Nrms_large_dataset.ipynb) applys the NRMS algorithm to the dataset that was used in the contest. This is what was required to be used when competing in the competition years ago.  Time to run on my setup: 15 hours and 36 minutes.

## SUBER Baseline

The [SUBER Baseline Jupyter Notebook](suber_baseline.ipynb) enables one to run the SUBER algorithm using LLMs in the original setup used from [Team SUBER](https://github.com/SUBER-Team). It is based on the paper: 

**"An LLM-based Recommender System Environment" by Nathan Coreco*, Giorgio Piatti*, Luca A. Lanzendörfer, Flint Xiaofeng Fan, Roger Wattenhofer.**

After working with Pydantic AI I am certain the approach can be improved.

