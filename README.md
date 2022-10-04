# bulk-labeling
A tool for quickly adding labels to unlabeled datasets

Running on [streamlit!](https://rungalileo-bulk-labeling-app-0l2mzc.streamlitapp.com/)

## How to use
We can walk through a simple example of going from an unlabeled dataset to some usable labels in just a few minutes

First, go to the streamlit app above, or you can [run it locally]((#Run-locally))

Then upload a csv file with your text. The only requirement of the file is that it must have a `text` column. Any other columns added can be used for coloring the embedding plot

[img]

Once the embeddings have processed, you'll see your dataframe on the left and embeddings on the right.

[img]

You can filter with the text search (regex coming soon!) or, by lasso selecting embedding clusters from the chart.

[img]

Since we see some clear clusters already, let's start by investigating them. We can see one cluster with a lot of references to weather.
Let's select this cluster

[video of hovering over weather samples, then selecting]

Confirming that this is about weather, we can register a new label "weather" and assign our samples

[video of creating label and assigning]

The UI will reset automatically. Let's look at another one. This cluster has a lot of references to bookings and reservations. Let's select that one.

[video of selecting reservations]

We can use the table's builtin text search to see how many references to "book" there are without actually filtering the table

[image of table search for book]

Let's register a new label "reservations" and label these samples

[video of reservations label]

We can inspect our labeled samples in the label-viewer page.

[img pointing to label viewer] [image of label-viewer page]

Once we are ready, we simple click "Export assigned labels" and then click the "Download" button

[image of exporting]

We just labeled N samples in a few minutes!





## Run locally

If you have a GPU running locally, want to try different encoder algorithms, or don't want to upload your data, you can run this locally.

1. Create a virtual environment (I recommend pyenv)
```
pyenv install $(cat .python-version)
python -m venv .venv
source .venv/bin/activate
# Check that it worked
which python pip
```
2. Install reqs `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
