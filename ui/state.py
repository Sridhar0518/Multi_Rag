import streamlit as st

from config import Hyperparams, get_llm
from index.embeddings import BgeEmbedder
from index.graph_store import NetworkXGraph
from index.vector_store import VectorIndex
from rag.common import RagContext


@st.cache_resource(show_spinner="Loading bge-large-en-v1.5 embedder...")
def get_embedder():
    return BgeEmbedder()


def get_ctx() -> RagContext:
    if "ctx" not in st.session_state:
        st.session_state.ctx = RagContext(VectorIndex(get_embedder()), NetworkXGraph(), get_llm(), Hyperparams())
        st.session_state.disabled_nodes = []
        st.session_state.disabled_edges = []
        st.session_state.history = []
        st.session_state.last = None
    return st.session_state.ctx
