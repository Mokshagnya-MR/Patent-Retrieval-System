"""Streamlit front-end for the Patent IR API.

Run from the repo root (so .streamlit/config.toml's theme is picked up):
    uvicorn src.api.main:app          # terminal 1
    streamlit run ui/streamlit_app.py # terminal 2
Set PATENT_IR_API to point at a non-default API address.
"""
import html
import os
import re
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("PATENT_IR_API", "http://127.0.0.1:8000")
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TIMEOUT_S = 120

METHODS = {
    "hybrid": "Hybrid: keywords + meaning",
    "bm25": "BM25: keyword match",
    "bm25f": "BM25F: field-weighted keywords",
    "semantic": "Semantic: embedding similarity",
}
CLASS_INFO = {"H01M": ("Batteries", "#1BAF7A"), "G06N": ("AI / ML", "#4A3AA7")}
MODE_TEXT, MODE_PATENT = "Describe an invention", "Start from a patent"
EXAMPLE_QUERIES = [
    "Sulfide solid electrolyte for an all-solid-state lithium battery",
    "Neuromorphic chip using memristor crossbar arrays",
    "Air cooling of a fuel cell stack in a vehicle",
    "Detecting anomalies in network traffic with machine learning",
]
EXAMPLE_PATENTS = {"10521475": "Travel-related cognitive profiles", "9172112": "Sulfide solid electrolyte glass",
                   "10062031": "Cognitive profiles"}

st.set_page_config(page_title="Prior-art search", page_icon="", layout="wide")

st.markdown("""
<style>
.block-container { max-width: 1080px; padding-top: 2.2rem; }
h1 { font-weight: 600; letter-spacing: -0.01em; }
.pa-lede { color: #4A5568; font-size: 1.02rem; max-width: 68ch; margin: -0.4rem 0 1.2rem; }
.pa-summary { color: #4A5568; margin: 0.4rem 0 0.8rem; }
.pa-head { display: flex; align-items: baseline; gap: 0.9rem; flex-wrap: wrap; }
.pa-rank { color: #7A8494; font-variant-numeric: tabular-nums; min-width: 1.6rem; }
.pa-number { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.45rem; font-weight: 600;
             color: #14213D; font-variant-numeric: lining-nums tabular-nums; letter-spacing: 0.01em; }
.pa-meta { color: #5B6474; font-size: 0.88rem; display: flex; gap: 1rem; flex-wrap: wrap; margin-left: auto; }
.pa-dot { display: inline-block; width: 0.6rem; height: 0.6rem; border-radius: 50%; margin-right: 0.35rem;
          vertical-align: 0.05rem; }
.pa-title { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.12rem; line-height: 1.35;
            color: #14213D; margin: 0.25rem 0 0.45rem 2.5rem; max-width: 70ch; }
.pa-abstract { color: #2D3748; line-height: 1.55; margin: 0 0 0.3rem 2.5rem; max-width: 76ch; font-size: 0.95rem; }
.pa-why { margin: 0.6rem 0 0.1rem 2.5rem; max-width: 76ch; }
.pa-why p { margin: 0.1rem 0 0.5rem; padding-left: 0.8rem; border-left: 2px solid #D5DBE3;
            color: #2D3748; font-size: 0.92rem; line-height: 1.5; }
.pa-label { color: #5B6474; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.2rem; }
.pa-family { margin: 0.5rem 0 0 2.5rem; color: #5B6474; font-size: 0.88rem; }
mark { background: #FFE36E; color: inherit; padding: 0 0.12em; border-radius: 2px; }
.pa-empty { padding: 2rem 0; color: #5B6474; max-width: 60ch; }
.pa-stat { font-family: 'Source Serif 4', Georgia, serif; font-size: 2rem; font-weight: 600; color: #14213D;
           font-variant-numeric: lining-nums tabular-nums; line-height: 1.1; }
.pa-stat-label { color: #5B6474; font-size: 0.88rem; }
@media (max-width: 640px) {
  .pa-title, .pa-abstract, .pa-why, .pa-family { margin-left: 0; }
  .pa-meta { margin-left: 0; }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers

def fmt_number(pid: str) -> str:
    return f"US {int(pid):,}" if pid.isdigit() else pid


def fmt_date(iso: str | None) -> str:
    if not iso:
        return "unknown"
    d = date.fromisoformat(iso)
    return f"{d.day} {d.strftime('%b %Y')}"


def readable_title(title: str) -> str:
    """Many HUPD titles are ALL CAPS; render those in sentence case."""
    return title[:1] + title[1:].lower() if title.isupper() else title


def highlight(text: str, terms: list[str]) -> str:
    """HTML-escape `text`, then wrap whole-word occurrences of query terms in <mark>."""
    escaped = html.escape(text)
    if not terms:
        return escaped
    alternation = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.sub(rf"(?<![A-Za-z0-9])({alternation})(?![A-Za-z0-9])", r"<mark>\1</mark>", escaped,
                  flags=re.IGNORECASE)


def truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


class ApiError(Exception):
    pass


def api(method: str, path: str, **kwargs) -> dict:
    try:
        resp = requests.request(method, f"{API_BASE}{path}", timeout=TIMEOUT_S, **kwargs)
    except requests.exceptions.ConnectionError:
        raise ApiError(f"Can't reach the search API at {API_BASE}. Start it from the repo root with "
                       "`uvicorn src.api.main:app`, then try again.")
    except requests.exceptions.Timeout:
        raise ApiError(f"The search API didn't respond within {TIMEOUT_S} seconds.")
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        if isinstance(detail, list):  # pydantic validation errors
            detail = "; ".join(d.get("msg", str(d)) for d in detail)
        raise ApiError(str(detail))
    return resp.json()


@st.cache_data(ttl=30, show_spinner=False)
def _cached_health() -> dict:
    return api("GET", "/health")  # raises on failure, and exceptions aren't cached


def api_health() -> dict | None:
    try:
        return _cached_health()
    except ApiError:
        return None


@st.cache_data(show_spinner=False)
def patent_details(pid: str) -> dict:
    return api("GET", f"/patent/{pid}")


# ---------------------------------------------------------------- state

defaults = {"mode": MODE_TEXT, "query": "", "patent_input": "", "response": None, "error": None,
            "run_pending": False}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)
if st.session_state.mode is None:  # segmented_control allows deselecting; must reset before it renders
    st.session_state.mode = MODE_TEXT


def queue_similar(pid: str):
    st.session_state.mode = MODE_PATENT
    st.session_state.patent_input = pid
    st.session_state.run_pending = True


def use_example_query():
    if st.session_state.get("example_query"):
        st.session_state.query = st.session_state.example_query
        st.session_state.example_query = None
        st.session_state.run_pending = True


def use_example_patent():
    if st.session_state.get("example_patent"):
        st.session_state.patent_input = st.session_state.example_patent.split(" ")[0]
        st.session_state.example_patent = None
        st.session_state.run_pending = True


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.subheader("Search settings")
    method = st.radio("Ranking method", list(METHODS), format_func=METHODS.get, key="method",
                      help="Hybrid scored best in our evaluation (NDCG@10 0.236 vs 0.211 for BM25). "
                           "See the Evaluation tab.")
    k = st.slider("Results to show", 5, 50, 10, step=5, key="k")
    diversify = st.toggle("Group near-duplicate patents", value=True, key="diversify",
                          help="Patents from the same family often have near-identical text. Grouping keeps the "
                               "top-ranked one and lists the rest underneath it.")
    st.divider()
    health = api_health()
    if health is None:
        st.error(f"Search API offline at {API_BASE}")
        st.caption("Start it with `uvicorn src.api.main:app` from the repo root.")
    elif health["status"] != "ok":
        st.warning("API is running but some index files are missing: " + ", ".join(health["missing"]))
    else:
        st.caption(f"Connected to the search API. {health['n_patents']:,} patents indexed, "
                   f"filed 2012–2018 in CPC classes H01M and G06N.")


# ---------------------------------------------------------------- page

st.title("Prior-art search")
st.markdown('<p class="pa-lede">Find earlier US patents related to an invention in batteries (H01M) or '
            'AI and machine learning (G06N). Results show the passages that matched, so you can judge '
            'relevance yourself. This tool does not assess patentability or novelty.</p>',
            unsafe_allow_html=True)

tab_search, tab_eval, tab_about = st.tabs(["Search", "Evaluation", "How it works"])


def run_search():
    st.session_state.error = None
    try:
        with st.spinner("Searching 14,008 patents…"):
            if st.session_state.mode == MODE_TEXT:
                query = st.session_state.query.strip()
                if not query:
                    st.session_state.error = "Describe the invention first, in a sentence or a paragraph."
                    return
                body = {"query": query, "k": k, "method": method, "diversify": diversify}
                if st.session_state.get("use_cutoff") and st.session_state.get("cutoff_date"):
                    body["filed_before"] = st.session_state.cutoff_date.isoformat()
                resp = api("POST", "/search", json=body)
                resp["_source"] = None
            else:
                pid = re.sub(r"[^0-9]", "", st.session_state.patent_input)
                if not pid:
                    st.session_state.error = "Enter a US patent number, like 9172112."
                    return
                params = {"k": k, "method": method, "diversify": diversify,
                          "prior_art_only": st.session_state.get("prior_art_only", True)}
                resp = api("POST", f"/similar/{pid}", params=params)
                resp["_source"] = patent_details(pid)
            st.session_state.response = resp
    except ApiError as e:
        st.session_state.error = str(e)
        st.session_state.response = None


@st.dialog("Patent details", width="large")
def show_details(pid: str):
    try:
        p = patent_details(pid)
    except ApiError as e:
        st.error(str(e))
        return
    st.markdown(f'<div class="pa-number">{fmt_number(pid)}</div>'
                f'<div class="pa-title" style="margin-left:0">{html.escape(readable_title(p["title"]))}</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Filed**  \n{fmt_date(p['filing_date'])}")
    c2.markdown(f"**Published**  \n{fmt_date(p['publication_date'])}")
    c3.markdown(f"**Main CPC**  \n{p['main_cpc_label'] or 'unknown'}")
    st.markdown("**Abstract**")
    st.write(p["abstract"])
    if p["first_claim"]:
        st.markdown(f"**First claim** (of {p['n_claims']})")
        st.write(p["first_claim"])
    if p["inventors"]:
        def cap(n: str) -> str:
            return n.title() if n.isupper() else n
        names = [f"{cap(i.get('inventor_name_first', ''))} {cap(i.get('inventor_name_last', ''))}".strip()
                 for i in p["inventors"]]
        st.markdown("**Inventors**  \n" + ", ".join(names))
    st.caption("CPC codes: " + ", ".join(p["cpc_codes"]))
    st.link_button("Open on Google Patents", f"https://patents.google.com/patent/US{pid}")


def render_result(r: dict, terms: list[str]):
    cls_name, cls_color = CLASS_INFO.get(r.get("cpc_class") or "", ("Other", "#7A8494"))
    passages = "".join(f"<p>{highlight(truncate(p, 420), terms)}</p>" for p in r["highlighted_passages"])
    family = ""
    if r["family_members"]:
        members = ", ".join(fmt_number(m) for m in r["family_members"])
        noun = "patent" if len(r["family_members"]) == 1 else "patents"
        family = f'<div class="pa-family">Grouped with {len(r["family_members"])} near-duplicate {noun}: {members}</div>'
    why = f'<div class="pa-why"><div class="pa-label">Best-matching claim passages</div>{passages}</div>' if passages else ""
    with st.container(border=True):
        st.markdown(f"""
<div class="pa-head">
  <span class="pa-rank">{r['rank']}</span>
  <span class="pa-number">{fmt_number(r['patent_id'])}</span>
  <span class="pa-meta">
    <span><span class="pa-dot" style="background:{cls_color}"></span>{cls_name} ({r.get('cpc_class') or 'other'})</span>
    <span>Filed {fmt_date(r.get('filing_date'))}</span>
    <span title="Retrieval score; its scale depends on the ranking method">Score {r['score']:.3f}</span>
  </span>
</div>
<div class="pa-title">{html.escape(readable_title(r['title']))}</div>
<div class="pa-abstract">{highlight(truncate(r['abstract'], 480), terms)}</div>
{why}
{family}
""", unsafe_allow_html=True)
        b1, b2, b3, _ = st.columns([1.2, 1, 1.3, 2.5])
        b1.button("Find similar", key=f"sim_{r['patent_id']}", on_click=queue_similar,
                  args=(r["patent_id"],), width="stretch")
        if b2.button("Details", key=f"det_{r['patent_id']}", width="stretch"):
            show_details(r["patent_id"])
        b3.link_button("Google Patents", f"https://patents.google.com/patent/US{r['patent_id']}",
                       width="stretch")


with tab_search:
    st.segmented_control("Search by", [MODE_TEXT, MODE_PATENT], key="mode", label_visibility="collapsed")

    with st.form("search_form", border=False):
        if st.session_state.mode == MODE_TEXT:
            st.text_area("Describe the invention", key="query", height=110,
                         placeholder="e.g. A cathode coating that suppresses dendrite growth in lithium-metal cells")
            c1, c2 = st.columns([1, 2])
            c1.checkbox("Only patents filed before", key="use_cutoff")
            c2.date_input("Cutoff date", key="cutoff_date", value=date(2016, 1, 1), min_value=date(2000, 1, 1),
                          max_value=date(2019, 12, 31), label_visibility="collapsed")
        else:
            st.text_input("US patent number", key="patent_input", placeholder="e.g. 9172112")
            st.checkbox("Only show patents filed before this one (prior art)", key="prior_art_only", value=True,
                        help="A patent filed later can't be prior art. Without this filter about a third of the "
                             "top 10 results are later filings.")
        submitted = st.form_submit_button("Search patents", type="primary")

    if st.session_state.mode == MODE_TEXT:
        st.pills("Try an example", EXAMPLE_QUERIES, key="example_query", on_change=use_example_query)
    else:
        st.pills("Try an example", [f"{pid} ({name})" for pid, name in EXAMPLE_PATENTS.items()],
                 key="example_patent", on_change=use_example_patent)

    if submitted or st.session_state.run_pending:
        st.session_state.run_pending = False
        run_search()

    if st.session_state.error:
        st.error(st.session_state.error)

    resp = st.session_state.response
    if resp is None and not st.session_state.error:
        st.markdown('<div class="pa-empty">Describe an invention in plain language, or enter a patent number to '
                    'find earlier patents like it. Results come from 14,008 granted US patents.</div>',
                    unsafe_allow_html=True)
    elif resp is not None:
        src = resp.get("_source")
        if src:
            st.markdown(f"Patents similar to **{fmt_number(src['patent_id'])}**, "
                        f"*{html.escape(readable_title(src['title']))}* (filed {fmt_date(src['filing_date'])}).")
        results = resp["results"]
        method_name = METHODS[resp["method"]].split(":")[0]
        alpha_note = f" (BM25 weight α = {resp['alpha']:.1f})" if resp.get("alpha") is not None else ""
        parts = [f"{len(results)} results in {resp['latency_ms']:.0f} ms using {method_name}{alpha_note}."]
        if resp.get("filed_before"):
            if resp["n_excluded_later_filed"]:
                parts.append(f"Hid {resp['n_excluded_later_filed']} patents filed after "
                             f"{fmt_date(resp['filed_before'])}.")
            else:
                parts.append(f"Showing only patents filed on or before {fmt_date(resp['filed_before'])}.")
        if resp["n_collapsed_duplicates"]:
            parts.append(f"Grouped {resp['n_collapsed_duplicates']} near-duplicates.")
        st.markdown(f'<div class="pa-summary">{" ".join(parts)}</div>', unsafe_allow_html=True)

        if not results:
            st.info("No patents matched. Try a longer description, a different ranking method, "
                    "or a later cutoff date.")
        for r in results:
            render_result(r, resp.get("query_terms", []))

        if results:
            export = pd.DataFrame([{
                "rank": r["rank"], "patent_number": r["patent_id"], "title": r["title"],
                "filing_date": r["filing_date"], "cpc_class": r["cpc_class"], "score": r["score"],
                "grouped_duplicates": " ".join(r["family_members"]),
            } for r in results])
            st.download_button("Download results as CSV", export.to_csv(index=False).encode("utf-8"),
                               file_name="prior_art_results.csv", mime="text/csv")


# ---------------------------------------------------------------- evaluation tab

def read_csv(name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


AXIS_STYLE = dict(labelColor="#4A5568", titleColor="#4A5568", domainColor="#C9D1DB",
                  labelFont="Public Sans", titleFont="Public Sans")


def stat_row(stats: list[tuple[str, str]]):
    for col, (value, label) in zip(st.columns(len(stats)), stats):
        col.markdown(f'<div class="pa-stat">{value}</div><div class="pa-stat-label">{label}</div>',
                     unsafe_allow_html=True)


with tab_eval:
    final = read_csv("final_comparison_table.csv")
    temporal = read_csv("temporal_filter.csv")
    if final is None:
        st.info("No evaluation results yet. Run `python -m src.eval.final_comparison` to create them.")
    else:
        st.markdown("All methods are scored on the same 62 held-out test queries. Each query is a granted "
                    "patent; the patents it cites (within this corpus) count as the relevant results. "
                    "Tuning, such as the hybrid weight α, used only the separate 144-query dev split.")

        if temporal is not None:
            best = temporal.loc[temporal["NDCG@10_filtered"].idxmax()]
            stat_row([
                (f"{best['NDCG@10_filtered']:.3f}", f"Best NDCG@10: {best['Method']} with later filings removed"),
                (f"{temporal['later_filed_in_top10'].mean():.1f} of 10",
                 "Top results filed after the query patent, averaged over methods"),
                (f"+{temporal['NDCG@10_rel_change'].median():.0%}",
                 "Median NDCG@10 gain from removing later filings"),
            ])

            st.subheader("NDCG@10 by method")
            long = temporal.melt(id_vars="Method", value_vars=["NDCG@10", "NDCG@10_filtered"],
                                 var_name="Ranking", value_name="NDCG@10 score")
            long["Ranking"] = long["Ranking"].map({"NDCG@10": "As ranked",
                                                   "NDCG@10_filtered": "Later filings removed"})
            series = ["As ranked", "Later filings removed"]
            chart = (
                alt.Chart(long)
                .mark_bar(cornerRadiusEnd=4, height=11)
                .encode(
                    y=alt.Y("Method:N", sort=temporal["Method"].tolist(), title=None,
                            axis=alt.Axis(labelLimit=220)),
                    yOffset=alt.YOffset("Ranking:N", sort=series),
                    x=alt.X("NDCG@10 score:Q", title="NDCG@10", scale=alt.Scale(domain=[0, 0.3]),
                            axis=alt.Axis(grid=True, gridColor="#E3E8EE", tickCount=6)),
                    color=alt.Color("Ranking:N", sort=series, scale=alt.Scale(domain=series,
                                                                              range=["#2A78D6", "#EB6834"]),
                                    legend=alt.Legend(orient="top", title=None, labelFont="Public Sans",
                                                      labelColor="#14213D")),
                    tooltip=["Method", "Ranking", alt.Tooltip("NDCG@10 score:Q", format=".4f")],
                )
                .properties(height=alt.Step(26))
                .configure_view(stroke=None)
                .configure_axis(**AXIS_STYLE)
            )
            st.altair_chart(chart, width="stretch")
            st.caption("Removing later-filed patents is significant for every method except BM25 "
                       "(Wilcoxon signed-rank, Holm-adjusted p ≈ 0.025). It makes 11 of 216 test citations "
                       "unreachable, because some patents cite later-filed family members.")
            with st.expander("Table view"):
                st.dataframe(temporal, hide_index=True, width="stretch")

        st.subheader("Full comparison, as ranked")
        show = final.rename(columns={"vs_prev_p_value": "p vs. previous row", "vs_prev_p_holm": "Holm-adjusted p"})
        st.dataframe(show, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f")
                                    for c in show.columns if c != "Method"})
        st.caption("p-values come from paired Wilcoxon signed-rank tests on per-query NDCG@10, each row against "
                   "the one above it. After Holm correction for the five tests, none of the step-to-step "
                   "differences is significant at 0.05.")

        cv = read_csv("cv_hybrid.csv")
        if cv is not None:
            st.subheader("All 206 queries, 5-fold cross-validation")
            st.markdown("62 test queries are too few to separate methods that differ by 0.02–0.03 NDCG@10. "
                        "Here α is picked on four folds and applied to the fifth, so every query is scored once "
                        "as held-out. α = 0.5 won in every fold.")
            methods = cv[cv["row_type"] == "method"][["name", "P@10", "R@10", "MRR", "NDCG@10"]]
            st.dataframe(methods.rename(columns={"name": "Method"}), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.4f")
                                        for c in ["P@10", "R@10", "MRR", "NDCG@10"]})
            tests = cv[cv["row_type"] == "test"][["name", "mean_a", "mean_b", "p_value", "p_value_holm"]]
            st.dataframe(tests.rename(columns={"name": "Comparison (NDCG@10)", "mean_a": "First",
                                               "mean_b": "Second", "p_value": "p", "p_value_holm": "Holm-adjusted p"}),
                         hide_index=True, width="stretch",
                         column_config={"First": st.column_config.NumberColumn(format="%.4f"),
                                        "Second": st.column_config.NumberColumn(format="%.4f"),
                                        "p": st.column_config.NumberColumn(format="%.2e"),
                                        "Holm-adjusted p": st.column_config.NumberColumn(format="%.2e")})

        sweep = read_csv("alpha_sweep.csv")
        if sweep is not None:
            st.subheader("Choosing the hybrid weight on the dev split")
            best_alpha = sweep.loc[sweep["ndcg_10"].idxmax()]
            base = alt.Chart(sweep).encode(
                x=alt.X("alpha:Q", title="α (0 = embeddings only, 1 = BM25 only)",
                        axis=alt.Axis(values=[i / 10 for i in range(11)], format=".1f", grid=False)),
                y=alt.Y("ndcg_10:Q", title="Dev NDCG@10", scale=alt.Scale(zero=False),
                        axis=alt.Axis(gridColor="#E3E8EE", tickCount=5)),
            )
            hover = alt.selection_point(on="pointerover", nearest=True, fields=["alpha"], empty=False)
            line = base.mark_line(color="#2A78D6", strokeWidth=2)
            points = base.mark_point(color="#2A78D6", filled=True, size=70).encode(
                opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                tooltip=[alt.Tooltip("alpha:Q", format=".1f", title="α"),
                         alt.Tooltip("ndcg_10:Q", format=".4f", title="NDCG@10"),
                         alt.Tooltip("mrr:Q", format=".4f", title="MRR")],
            ).add_params(hover)
            best_df = pd.DataFrame([best_alpha])
            best_pt = alt.Chart(best_df).mark_point(color="#2A78D6", filled=True, size=90, stroke="#FFFFFF",
                                                    strokeWidth=2, opacity=1).encode(x="alpha:Q", y="ndcg_10:Q")
            label = alt.Chart(best_df).mark_text(dy=-14, color="#14213D", font="Public Sans").encode(
                x="alpha:Q", y="ndcg_10:Q", text=alt.value(f"best α = {best_alpha['alpha']:.1f}"))
            st.altair_chart((line + points + best_pt + label).properties(height=260)
                            .configure_view(stroke=None).configure_axis(**AXIS_STYLE),
                            width="stretch")

        qr_tfidf, qr_simple = read_csv("query_reduction_ablation.csv"), read_csv("query_reduction_ablation_simple.csv")
        if qr_tfidf is not None and qr_simple is not None:
            st.subheader("How much of a patent to use as the query")
            rows = []
            for df, reduced_name in [(qr_tfidf, "Top 40 TF-IDF terms"), (qr_simple, "First claim + abstract")]:
                for _, r in df.iterrows():
                    source = ("Full text (title, abstract, claims, background)" if r["query_mode"] == "patent_full"
                              else reduced_name)
                    rows.append({"Query built from": source, "Method": "BM25" if r["method"] == "bm25" else "Hybrid",
                                 "NDCG@10": r["ndcg_10"], "P@10": r["p_10"], "R@10": r["recall_10"]})
            qr = pd.DataFrame(rows).drop_duplicates(["Query built from", "Method"])
            st.dataframe(qr, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.4f")
                                        for c in ["NDCG@10", "P@10", "R@10"]})
            st.caption("Scored on all 206 queries with the full qrels, so these numbers aren't comparable to the "
                       "test-split tables above.")

        ann = read_csv("ann_tradeoff.csv")
        if ann is not None:
            st.subheader("Exact vs. approximate vector search")
            recall_col = next(c for c in ann.columns if c.startswith("ivf_recall"))
            stat_row([
                (f"{ann['flat_latency_ms'].mean():.2f} ms", "Exact search (FAISS flat), mean per query"),
                (f"{ann['ivf_latency_ms'].mean():.2f} ms", "Approximate search (FAISS IVF), mean per query"),
                (f"{ann[recall_col].mean():.0%}", "Share of the exact top 10 that IVF also returns"),
            ])


# ---------------------------------------------------------------- about tab

with tab_about:
    st.markdown("""
**What gets searched.** Title, abstract and claims of 14,008 granted US patents filed 2012–2018 in two CPC
classes: H01M (batteries and electrochemistry) and G06N (AI and machine learning), from the Harvard USPTO
Patent Dataset.

**How results are ranked.**
- *BM25* scores exact keyword overlap across title, abstract and claims.
- *BM25F* scores each field separately and weights claims twice as heavily.
- *Semantic* compares meaning using `bge-small-en-v1.5` embeddings of the title and abstract.
- *Hybrid* normalizes BM25 and semantic scores and blends them with weight α, chosen on dev queries only.

**What happens after ranking.**
1. The query patent itself is removed (in patent mode).
2. Patents filed after the cutoff are removed, if the cutoff is on.
3. Near-duplicates are grouped: results whose title+abstract TF-IDF similarity is 0.85 or more fold into
   the highest-ranked one.
4. Each result shows its two best-matching passages from the claims, ranked by overlap with your query's
   content words.

**Limits.** Relevance judgments come from patent citations, which are incomplete, so some results counted
as "not relevant" in evaluation are real prior art. The corpus covers two technology classes only. This is
a discovery aid. It does not assess patentability or novelty.
""")
