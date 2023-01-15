# bulk-labeling
A tool for quickly adding labels to unlabeled datasets

Running on [streamlit!](https://rungalileo-bulk-labeling-app-0l2mzc.streamlitapp.com/)

## How to use
We can walk through a simple example of going from an unlabeled dataset to some usable labels in just a few minutes

First, go to the streamlit app above, or you can [run it locally](#run-locally)

Then upload a csv file with your text. The only requirement of the file is that it must have a `text` column. Any other columns added can be used for coloring the embedding plot. If you don't have one, you can use the [conv-intent](https://github.com/rungalileo/bulk-labeling/blob/main/conv_intent.csv) dataset from this repo!

![image](https://user-images.githubusercontent.com/22605641/212553133-1cb5342c-5636-4b8b-bae6-e811b6186614.png)


Once the embeddings have processed, you'll see your dataframe on the left and embeddings on the right. The dataframe view comes with an extra `text_length` column that you can sort by, or color the embeddings plot with (in case text length is useful to you).

You can filter with the text search (regex coming soon!) or, by lasso selecting embedding clusters from the chart. You can also color the chart and resize the points using the menu on the left

![image](https://user-images.githubusercontent.com/22605641/193920464-bee6c734-6ad9-45cc-83e0-00dc5a27f4e4.png)


Since we see some clear clusters already, let's start by investigating them. We can see one cluster with a lot of references to weather.
Let's select this cluster

https://user-images.githubusercontent.com/22605641/193921160-b024c2f4-3057-41e6-a200-b73bf258e6e9.mov


Confirming that this is about weather, we can register a new label "weather" and assign our samples

https://user-images.githubusercontent.com/22605641/193921485-a052dfdb-e905-4860-b01b-202dce04486a.mov


The UI will reset automatically. Let's look at another one. This cluster has a lot of references to bookings and reservations. Let's select that one.

https://user-images.githubusercontent.com/22605641/193921981-6bd0c4a2-20d9-4de2-9c78-e0c334a6773c.mov


We can use the streamlit table's builtin text search (by clicking on the table, then CMD+F) to see how many references to "book" there are. Unlike the text search filter, this won't actually filter the selection.


https://user-images.githubusercontent.com/22605641/193922421-ea1940b4-00c9-40e8-969c-90aa84658ed0.mov



Loads of samples have "book" in them, but we can be a bit more generic and call this "reservations". Let's register a new label "reservations" and label these samples.


https://user-images.githubusercontent.com/22605641/193922719-cab4d3f2-7970-48df-87b0-4c39c82a8e75.mov


We can inspect our labeled samples in the label-viewer page.

![image](https://user-images.githubusercontent.com/22605641/193922845-34be5fa7-6803-4e1c-96c0-d0d4f9009032.png)

<img width="1680" alt="image" src="https://user-images.githubusercontent.com/22605641/193922962-751e8e58-df98-4d08-90f7-63dfc7d216ed.png">


Once we are ready, we simple click "Export assigned labels" and then click the "Download" button


https://user-images.githubusercontent.com/22605641/193927497-929b1e3a-f185-433c-a275-68c2972cdb91.mov


We just labeled N samples in a few minutes!

There are some pretty funny "mistakes" in the embeddings (samples that are semantically similar to other categories, but have words that trigger weather/reservation) that should be considered! The embeddings aren't perfect. We are using a smaller model (paraphrase-MiniLM-L3-v2) in order to get embeddings in a reasonable speed. But it's a good start! Feel free to run this locally and use a [better model](https://www.sbert.net/docs/pretrained_models.html)

<img width="854" alt="image" src="https://user-images.githubusercontent.com/22605641/193925517-5638a8f6-29c3-4023-9463-77ba92a89ffc.png">



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
2. Install reqs `pip install -r requirements.txt && pyenv rehash`
3. Run the app: `streamlit run app.py`
