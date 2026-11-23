import json

import pandas as pd
import streamlit as st

from config import CACHE_DIR, ROOT
from eval.benchmark import run_benchmark, summarize
from eval.metrics import cross_doc_resolution, entity_resolution_accuracy
from ui.state import get_ctx

ENGS = ["Standard RAG", "HyDE RAG", "Graph RAG"]


def render():
    ctx = get_ctx()
    st.subheader("Comparative benchmark (measured)")
    qa = json.loads((ROOT / "data" / "qa.json").read_text())
    judge = st.checkbox("Run faithfulness LLM-judge (extra LLM calls)", value=True)
    n = st.slider("Questions", 1, len(qa), len(qa))
    if st.button("Run benchmark on indexed corpus", disabled=not ctx.index.docs):
        bar = st.progress(0.0)
        rows = run_benchmark(ctx, qa[:n], judge, progress=lambda f, q, e: bar.progress(f, text=f"{q} / {e}"))
        gold = json.loads((ROOT / "data" / "entity_gold.json").read_text())
        summ = summarize(rows)
        summ["entity_resolution_accuracy"] = entity_resolution_accuracy(gold["alias_pairs"])
        summ["cross_doc_resolution"] = cross_doc_resolution(ctx.graph, gold["cross_doc"])
        (CACHE_DIR / "benchmark.json").write_text(json.dumps({"rows": rows, "summary": summ}, indent=1))
        bar.empty()
    if not ctx.index.docs:
        st.info("Load the sample corpus in Tab 1 to run the benchmark.")
    f = CACHE_DIR / "benchmark.json"
    if not f.exists():
        return
    data = json.loads(f.read_text())
    s = data["summary"]
    table = []
    for grp in ["all", "keyword", "multihop", "vocab"]:
        for eng in ENGS:
            v = s.get(f"{eng}|{grp}")
            if v:
                table.append({"slice": grp, "engine": eng, "RCR": round(v["rcr"], 2),
                              "faithfulness": round(v.get("faith", float("nan")), 2), "avg latency ms": round(v["total_ms"])})
    df = pd.DataFrame(table)
    st.dataframe(df, width="stretch", hide_index=True)
    st.bar_chart(df[df.slice == "all"].set_index("engine")[["RCR", "faithfulness"]])
    st.caption("Measured on a small synthetic corpus; results illustrate the method, not production performance.")
    with st.expander("Per-question rows"):
        st.dataframe(pd.DataFrame(data["rows"]).drop(columns=["answer"]))
