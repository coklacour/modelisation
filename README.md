## Configuration guidelines

### Python environement

* to facilitate collaboration, I've integrated the Poetry tool. Poetry is a dependency management and packaging tool in Python. for more info https://python-poetry.org/docs/

* To install poetry, go to this web page : https://python-poetry.org/docs/#installation

* To activate and configure the Poetry environment, you need to run the 2 commands below.

```bash
poetry shell
poetry install
```

* It is important to run these commands at the start of each session 

### About the project structure

To start this magnificent project, I propose the following structure (https://drivendata.github.io/cookiecutter-data-science/):

    .
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering)
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │            
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py