# bulk-labeling
A tool for quickly adding labels to unlabeled datasets

Running on [streamlit!](https://rungalileo-bulk-labeling-app-0l2mzc.streamlitapp.com/)

## Run locally

1. Create a virtual environment (I reccomend venv)
```
pyenv install $(cat .python-version)
python -m venv .venv
source .venv/bin/activate
# Check that it worked
which python pip
```
2. Install reqs `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
