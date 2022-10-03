import sys

import streamlit as st

from app import SessionKey, get_export_df

st.set_page_config(layout="wide")
st.header("🔎 Export Viewer")

df = st.session_state.get(SessionKey.df, [])
if len(df):
    df = get_export_df()
    st.dataframe(df)
else:
    st.subheader("Label samples to see them here!")
