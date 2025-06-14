### Toxicity Detector (PEFT + Civil Comments)

This project fine-tunes a lightweight BERT model using **LoRA (Low-Rank Adaptation)** via HuggingFace + PEFT to classify toxic comments from the [Civil Comments Dataset](https://huggingface.co/datasets/civil_comments). Designed to be resource-efficient and easy to run on laptops or servers with limited GPU access.


### Features

- Dataset: Civil Comments
- Lightweight model (DistilBERT) for speed
- Efficient fine-tuning using PEFT (LoRA)
- Dynamic device handling (CPU, CUDA, MPS)
- Modular training, inference, and evaluation scripts
- Jupyter notebook-ready

### Usage

src/training.py: training and saving trained file
src/evaluation.py: evaluate the save model using test data
notebooks/training.ipynb: jupyter notebook of training, with example data, etc.

### Setup

- Make sure you have Python ≥ 3.10.

pipenv shell
pipenv install torch transformers datasets peft accelerate
pipenv install ipykernel
python -m ipykernel install --user --name=toxic-env --display-name "Python (toxic-env)"
cd notebooks
jupyter notebook
