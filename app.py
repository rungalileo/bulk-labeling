import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from numerize.numerize import numerize
from sentence_transformers import SentenceTransformer
from streamlit.elements.file_uploader import UploadedFile
from streamlit_plotly_events import plotly_events
from umap import UMAP

sys.path.append(".")


@dataclass
class SessionKey:
    model: str = "model"
    figure_state: str = "figure_state"
    file: str = "file"
    df: str = "df"
    active_ids: str = "active_ids"
    fig: str = "fig"
    selected_points: str = "selected_points"
    has_xy: str = "has_xy"
    marker_size: str = "marker_size"
    color: str = "color"
    labels: str = "labels"
    label_assignments: str = "label_assignments"
    is_expanded: str = "is_expanded"
    default_index: str = "default_index"
    chosen_label: str = "chosen_label"
    label_select_key: str = "label_select_key"


@dataclass
class InternalCol:
    hovertext: str = "hovertext"
    x: str = "x"
    y: str = "y"


INTERNAL_COLS = [InternalCol.hovertext, InternalCol.x, InternalCol.y]


def get_export_df() -> pd.DataFrame:
    df2 = st.session_state[SessionKey.df].copy()
    id_label = st.session_state[SessionKey.label_assignments]
    df2["label"] = df2["id"].apply(lambda id_: id_label.get(id_, -1))
    df2 = df2[df2["label"] != -1]

    cols = [c for c in df2.columns if c not in INTERNAL_COLS]
    return df2[cols]


# os.environ["TOKENIZERS_PARALLELISM"] = "false"
umap_model = UMAP(n_neighbors=15, random_state=42, verbose=True)
st.set_page_config(layout="wide")
st.title("Laboratory 🧪")
col1, col2 = st.columns([3, 4])

# The registered labels by the user
if SessionKey.labels not in st.session_state:
    st.session_state[SessionKey.labels] = []

# The assigned labels {tuple of ids: label}
if SessionKey.label_assignments not in st.session_state:
    st.session_state[SessionKey.label_assignments] = {}

if SessionKey.label_select_key not in st.session_state:
    st.session_state[SessionKey.label_select_key] = uuid4()


# Pre download the model
if SessionKey.model not in st.session_state.keys():
    SentenceTransformer("paraphrase-MiniLM-L3-v2")
    st.session_state[SessionKey.model] = True


def reset_plotly_figure(force: bool = False) -> None:
    """Reload the plotly chart from scratch, remove all state

    We are using a tool called streamlit_plotly_events to capture and maintain selected
    points from a lasso select. The issue with the package is that there's no way
    (from what I can tell) to drop the state of the chart (the selected points).

    But sometimes (often) a user wants to refresh and remove the selected points. The
    package does come with a kwarg `key` which defines the `id` of the chart in case
    you want to have multiple to keep tabs of. So if we change the `id` of the chart,
    we can essentially refresh the chart and remove the selected points
    """
    # The first time we call this, there won't be a figure state, but all subsequent
    # times there will be, so we include the `force` param to opt into "re-clearing"
    # the chart
    if SessionKey.figure_state not in st.session_state or force:
        st.session_state[SessionKey.figure_state] = str(uuid4())


def clear_state() -> None:
    """Clear the global state.

    Either when a new file is uploaded, or when the user wants to "start over" on their
    work
    """
    for key in st.session_state.keys():
        # No reason to delete the model, we have it downloaded and it doesn't change
        if key != SessionKey.model:
            del st.session_state[key]


def reset_embeddings() -> None:
    """Reset the embeddings view to full dataframe

    Remove all global state that involves the embeddings or filters on the dataframe
    """
    for key in [SessionKey.selected_points, SessionKey.fig, SessionKey.active_ids]:
        if key in st.session_state:
            del st.session_state[key]
    reset_plotly_figure(force=True)
    st.experimental_rerun()


def get_dataframe_file() -> UploadedFile:
    file = st.sidebar.file_uploader(
        "Upload your CSV text file", type="csv", on_change=clear_state
    )
    if SessionKey.file in st.session_state.keys():
        return st.session_state[SessionKey.file]
    st.session_state[SessionKey.file] = file
    return file


def apply_emb_model(text_chunk: List[str]) -> np.ndarray:
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return model.encode(text_chunk)


@st.cache(allow_output_mutation=True)
def get_text_embeddings(texts: List[str]) -> np.ndarray:
    return apply_emb_model(texts)
    # embs = []
    # chunk_size = math.ceil(len(texts) / 10)
    # text_chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    #
    # with ProcessPoolExecutor(max_workers=10) as pool:
    #     for text_chunk in text_chunks:
    #         embs.append(pool.submit(apply_emb_model, text_chunk))
    #
    # embs = [i.result() for i in embs]
    # return np.concatenate(embs)


@st.cache(allow_output_mutation=True)
def get_umap_embeddings(embs: np.ndarray) -> np.ndarray:
    return umap_model.fit_transform(embs)


def add_umap_embeddings(df: pd.DataFrame, emb_xy: np.ndarray) -> pd.DataFrame:
    df["x"] = emb_xy[:, 0]
    df["y"] = emb_xy[:, 1]
    return df


def clear_state_after_export() -> None:
    num_samples_exported = len(st.session_state[SessionKey.label_assignments])
    st.info(f"Exported {num_samples_exported} labeled samples!", icon="ℹ️")
    clear_state()


def export_label_assignments() -> None:
    if SessionKey.df in st.session_state and len(st.session_state[SessionKey.df]):
        df2 = get_export_df()
        st.sidebar.download_button(
            f"Download {numerize(len(df2))} samples",
            df2.to_csv(index=False).encode("utf-8"),
            file_name="export.csv",
            mime="text/csv",
            on_click=clear_state_after_export,
        )


def assign_label() -> None:
    """Saves a given label with a list of IDs to apply to"""
    key = st.session_state[SessionKey.label_select_key]
    chosen_label = st.session_state[key]
    ids_key = st.session_state[SessionKey.active_ids]
    if chosen_label in st.session_state[SessionKey.labels]:
        print(f"Setting {len(ids_key)} label to {chosen_label}")
        for id_key in ids_key:
            st.session_state[SessionKey.label_assignments][id_key] = chosen_label

        st.session_state[SessionKey.default_index] = 0
        st.session_state[SessionKey.is_expanded] = False
        st.info(f"{len(ids_key)} samples labeled {chosen_label}", icon="ℹ️")
        reset_plotly_figure(force=True)
    st.session_state[SessionKey.label_select_key] = uuid4()


class Laboratory:
    def __init__(self) -> None:
        reset_plotly_figure()
        # On page refresh, we need to reload our stateful attributes via session state
        self.file = st.session_state.get(SessionKey.file)
        self.df = st.session_state.get(SessionKey.df)
        self.embs: np.ndarray = np.ndarray([])
        self.umap_xy: np.ndarray = np.ndarray([])
        self.selected_points: List[int] = []
        self.ids = st.session_state.get(SessionKey.active_ids)
        self.force_new_fig = False

        self.sidebar()

        # We create the scatterplot and then refresh the app, so that it's the
        # first thing rendered. We need to do this because of the way that plotly_events
        # works. It stores the selected samples from the lasso, and we need to first
        # get those points and then filter the dataframe/embeddings based on them.
        if SessionKey.fig in st.session_state:
            with col2:
                self.plot_figure()

        if self.file:
            with col1:
                self.dataframe()
            with col2:
                self.embeddings()
                self.create_figure()

    def sidebar(self) -> None:
        self.file = get_dataframe_file()
        new_label = st.sidebar.text_input("Register Label")
        if new_label and new_label not in st.session_state[SessionKey.labels]:
            st.session_state[SessionKey.labels].append(new_label)
        # all_labels = st.sidebar.empty()
        with st.sidebar.expander("Current Labels"):
            for label in st.session_state[SessionKey.labels]:
                st.write(label)
        assigned = st.session_state.get(SessionKey.label_assignments) or {}
        if st.sidebar.button(
            f"Export {len(assigned)} Assigned labels", disabled=not assigned
        ):
            export_label_assignments()

        st.sidebar.markdown("---")
        # We don't want to be able to filter the dataframe until its fully processed
        self.search_term = st.sidebar.text_input(
            "Text Search", disabled=not st.session_state.get(SessionKey.has_xy, False)
        )
        st.sidebar.markdown("---")
        if st.sidebar.button("Reset Selection"):
            # We want to clear all selected points as well as the figure, and rerun
            # the app. This will cause all lasso selections to go away and give us
            # a fresh embedding scatterplot
            print("exporting")
            reset_embeddings()

        default = st.session_state.get(SessionKey.marker_size, 2)
        st.session_state[SessionKey.marker_size] = st.sidebar.slider(
            "point size", min_value=1, max_value=20, value=default
        )
        color_by = ["<select>"]
        if SessionKey.df in st.session_state:
            df = st.session_state[SessionKey.df]
            color_by += [c for c in df.columns if c not in ("id", "text", "hovertext")]
        default_color = st.session_state.get(SessionKey.color) or "<select>"
        default_index = color_by.index(default_color)
        color = st.sidebar.selectbox("Color By", color_by, index=default_index)
        st.session_state[SessionKey.color] = None if color == "<select>" else color

    def dataframe(self) -> None:
        st.subheader("DataFrame")

        if SessionKey.df not in st.session_state and self.file is not None:
            self.df = pd.read_csv(self.file)
            self.df["id"] = self.df.index
            self.df["text_length"] = self.df["text"].str.len()
            self.df["hovertext"] = self.df.text.str.wrap(30).str.replace("\n", "<br>")
            # Checkpoint the df
            st.session_state[SessionKey.df] = self.df

        assert self.df is not None
        # Apply search
        self.df = self.df[
            self.df.apply(lambda row: self.search_term in row["text"], axis=1)
        ]
        if st.session_state.get(SessionKey.selected_points):
            filter_ids = [
                i["pointIndex"] for i in st.session_state[SessionKey.selected_points]
            ]
            self.df = self.df[self.df["id"].isin(filter_ids)]
            # If this is a new lasso selection (new filter_ids), then we want to
            # redraw and re-render the embedding scatter.

        self.ids = self.df["id"].tolist()
        # Checkpoint the filtered ids
        st.session_state[SessionKey.active_ids] = self.ids

        showcols = [c for c in self.df.columns if c not in ("hovertext", "Unnamed: 0")]
        st.write(f"({len(self.df)}) active rows")
        label_assigner = st.expander(
            "Set label for selection",
            expanded=st.session_state.get(SessionKey.is_expanded, False),
        )
        with label_assigner:
            avl_labels = ["<select>"] + st.session_state[SessionKey.labels]
            st.selectbox(
                "Choose Label",
                avl_labels,
                key=st.session_state[SessionKey.label_select_key],
                on_change=assign_label,
            )

        st.dataframe(self.df[showcols], height=800)

    def create_figure(self) -> None:
        p = px.scatter(
            self.df,
            x="x",
            y="y",
            color=st.session_state[SessionKey.color],
            hover_data=["hovertext"],
        )
        p.update_traces(marker_size=st.session_state[SessionKey.marker_size])
        # If there's no figure yet or it's changed, refresh and replot it
        if (
            SessionKey.fig not in st.session_state
            or st.session_state[SessionKey.fig] != p
        ):
            st.session_state[SessionKey.fig] = p
            print("Forcing refresh")
            st.experimental_rerun()

    def plot_figure(self) -> None:
        st.subheader("Embeddings")
        st.session_state[SessionKey.selected_points] = plotly_events(
            st.session_state[SessionKey.fig],
            select_event=True,
            override_height=800,
            key=st.session_state[SessionKey.figure_state],
        )

    def embeddings(self) -> None:
        # Only calculate the UMAP embeddings once for a given dataframe. If we've
        # already done it, save the `has_xy` state and don't recalculate
        if SessionKey.has_xy not in st.session_state and self.df is not None:
            progress = st.empty()
            progress.text("Getting embeddings for text")
            self.embs = get_text_embeddings(self.df.text.tolist())
            progress.text("Applying UMAP")
            self.umap_xy = get_umap_embeddings(self.embs)
            progress.text("")
            self.df = add_umap_embeddings(self.df, self.umap_xy)
            st.session_state[SessionKey.df] = self.df
            # Set so we don't have to recalculate this on every interaction with the app
            st.session_state[SessionKey.has_xy] = True


if __name__ == "__main__":
    Laboratory()
