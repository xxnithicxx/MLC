# Simple Project for testing Multi Label Classification

## Project Description

```
project-folder/
│
├── app.py                     # Streamlit UI code entry point
├── config.yaml                # Configuration file (e.g., paths, hyperparameters)
├── requirements.txt           # Dependencies for the project
├── README.md                  # Project documentation
├── data/                      # Directory to store datasets and preprocessed data
│   └── multilabel_modified/   # Dataset folder downloaded from Kaggle
│       ├── images/            # Images used in the project
│       └── multilabel_classification(6)-reduced_modified.csv  # CSV file with labels
├── src/                       # Source code for your models and data handling
│   ├── __init__.py            # Makes src a package
│   ├── data_loader.py         # Data loading and transformations
│   ├── models/                # Folder for model classes
│   │   ├── __init__.py        # Makes models a package
│   │   ├── c2ae.py            # C2AE implementation
│   │   ├── ml_decoder.py      # MLDecoder implementation
│   └── utils.py               # Utility functions (e.g., for visualization)
├── notebooks/                 # Jupyter notebooks for experimentation
│   └── data_analysis.ipynb    # Example notebook for EDA
├── checkpoints/               # Directory to store model checkpoints
│   └── best_model.pth         # Example model checkpoint
```

## Project Setup

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/meherunnesashraboni/multi-label-image-classification-dataset) and place it in the `data` folder and extract it.

```bash
git clone
pip install -r requirements.txt
streamlit run app.py
```