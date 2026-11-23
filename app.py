import streamlit as st

st.set_page_config(page_title="PwC India Advisory - Multi-RAG Benchmark", layout="wide")
st.title("Standard vs HyDE vs Graph RAG - Advisory Benchmark")
st.caption("Sample documents are SYNTHETIC and for demonstration only.")

from ui import tab1, tab2, tab3, tab4, tab5  # noqa: E402

t = st.tabs(["1. Ingestion & Chat", "2. Knowledge Graph", "3. LangGraph Workflow", "4. Hyperparameters & KPIs", "5. Benchmark"])
for tab, mod in zip(t, [tab1, tab2, tab3, tab4, tab5]):
    with tab:
        mod.render()
