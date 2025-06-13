### here's a recommended process to begin
pipenv shell
pipenv install torch transformers datasets peft accelerate
pipenv install ipykernel
python -m ipykernel install --user --name=toxic-env --display-name "Python (toxic-env)"
cd notebooks
jupyter notebook
http://localhost:8888/notebooks/Untitled.ipynb

