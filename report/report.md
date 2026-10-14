# AI-Powered Patent Retrieval & Similarity Search

*Information Retrieval course project. BM25/vector retrieval, indexing, and
evaluation are the graded core; the LLM/explanation layer is a
thin, clearly-labeled bonus on top and does not make or imply any
patentability/novelty determination.*

## 1. Problem Statement and Dataset Rationale

We build a prior-art discovery system over a domain-narrowed subset of the
Harvard USPTO Patent Dataset (HUPD): granted US patents in two CPC classes,
**H01M** (batteries/electrochemistry) and **G06N** (AI/ML/computing),
filed 2012–2018. A narrow domain gives cleaner relevance judgments and more
interpretable "why retrieved" explanations than a broad random sample across
all CPC classes would.

We verified against the full HUPD metadata index that filing years 2012–2018
capture 16,471 of the 16,494 total H01M+G06N granted patents across the
dataset's entire 2004–2018 span (99.9%) — 2004–2011 contributes only ~23
patents combined, so restricting to 2012–2018 avoids downloading 8 additional
multi-gigabyte archives for negligible corpus growth.

**Actual corpus size:** 14,008 patents (10,176 H01M / 3,833 G06N) — no
extraction loss; every CPC-matched, granted patent in the metadata index for
2012–2018 was successfully recovered from the per-year archives.

## 2. Ground-Truth Methodology: Citation-Derived Qrels

**Why not HUPD's own fields:** HUPD's 34-field schema (application_number,
title, decision, CPC/IPC labels, filing/issue dates, examiner info,
abstract/claims/background/summary/full_description) has no citation field.
Citations had to be sourced externally.

**Why not USPTO's own citation API/bulk data:** As of June 2026, USPTO folded
PatentsView into its Open Data Portal (data.uspto.gov) and gated bulk citation
downloads behind a USPTO.gov account with mandatory MFA; the legacy
`search.patentsview.org` API host was decommissioned entirely. This is an
account-creation dependency outside what could be resolved autonomously in
this session.

**Source used:** Google's Patents Public Data on BigQuery
(`patents-public-data.patents.publications`), which parses the same
USPTO front-page "References Cited" record PatentsView/ODP would serve —
same underlying facts, different pipe. We restrict to granted US patents
(kind codes B1/B2) on both the citing and cited side, since that's where the
References Cited list is authoritative. The `citation.category` field was
designed to let us grade relevance (2 for examiner-added `EXA` citations,
1 otherwise) — but the categories actually observed in this corpus slice are
only `APP` (722, applicant-submitted) and `SEA` (467, search-report-cited);
`EXA` never appears. Rather than force a grading distinction the data
doesn't support, qrels here come out effectively binary. Documented as a
finding, not engineered around.

A cited patent is only usable as a qrel if it's *also* inside our H01M/G06N
corpus — a citation to a patent outside the CPC subset can't be evaluated
against. Queries (citing patents) need at least 2 in-subset relevant docs to
be usable (the plan's original bar was 3, but a strictly-in-subset citation
graph over just 2 CPC classes is inherently sparse: 759 citing patents have
≥1 in-subset relevant doc, 206 have ≥2, only 90 have ≥3 — 2 was the bar that
cleared the ~150-usable-queries target without discarding real, usable
citation pairs). See `src/preprocess/fetch_citations.py` and
`src/preprocess/build_qrels.py`.

**Limitations (explicit, as this is a proxy for true relevance, not
human judgment):**
- Citations are added during prosecution for legal/prior-art-disclosure
  reasons, not as a topical-similarity judgment — a cited patent may be
  only narrowly relevant to one claim element, not the invention as a whole.
- Citation coverage is inherently incomplete: a genuinely relevant
  prior-art patent that nobody happened to cite produces a false negative
  in our qrels, understating true recall for every method equally (a bias,
  not fixable within this ground-truth methodology, but consistent across
  all systems compared).
- Restricting to in-subset citations discards the majority of any given
  patent's real citations (most cited patents fall outside our 2-CPC-class
  slice) — our qrels measure retrieval quality only within the corpus we
  built, not against the true universe of prior art.

**Usable topic set:** 206 queries (637 qrels lines), split 70/30 dev/test
(`src/eval/splits.py`, seed=42) → 144 dev / 62 test. All tuning (alpha) is
done on dev only; the final comparison table below scores every method
against the same 62 held-out test queries (`data/qrels/qrels_test.txt`) so
comparisons are apples-to-apples even though the alpha-weighted hybrid run
only covers test-split queries by construction.

## 3. Methodology Per Stage

**A methodology bug was found and fixed mid-project, and every number below
reflects the corrected pipeline.** Every patent-as-query topic's query text
IS that patent's own title+abstract, so early runs had that same patent
trivially self-matching at rank 1 with a near-perfect score every time —
wasting a top-1 slot on a guaranteed non-relevant hit and deflating every
rank-sensitive metric uniformly. This was already handled correctly in the
API's `/similar` endpoint but missed in the evaluation pipeline. Caught by
noticing every top-1 result across a dozen manually-inspected examples was
the query patent itself (see Section 8), verified by filtering existing run
files before touching any code (NDCG@10 jumped 25-30% relative once
self-matches were excluded — too large to be cosmetic), then fixed at the
source (`exclude_self_match()` in `src/eval/trec_format.py`, applied in
`baseline.py` and `semantic.py`'s run generation) and the entire downstream
pipeline regenerated. Relative ordering between methods was largely
unaffected (self-matches hurt every method equally), but two conclusions
changed materially: BM25 vs BM25F crossed the unadjusted p<0.05 line
(it does not survive multiple-comparison correction; see Section 4),
and the query-reduction ablation's ranking of which reduction strategy hurts
least flipped (Section 5).

**Fixes from a later code review.** A second review pass found four more
defects. None changes the Section 4 table, but one invalidated an ablation
row (Section 5):
1. *Claim-1 extraction never split claims.* `extract_claim_1` split on
   newline-plus-number, but HUPD stores every patent's claims on a single
   line (`"1-15. (canceled) 16. A quantum circuit ... 17. The circuit of
   claim 16 ..."`). It returned the *entire* claims field for 100% of the
   corpus, so the "claim 1 + abstract" query variant was really "all claims
   + abstract". Rewritten as an inline, sequence-validated splitter that
   also skips canceled claim ranges (`split_claims` in
   `src/retrieve/query_reduction.py`). Claim 1 is now a median 14% of the
   claims text, and the ablation was re-run.
2. *Non-deterministic tie order in weighted fusion.* Candidates were
   collected in a Python `set` of doc-id strings, whose iteration order
   changes on every run (hash randomization). Tied scores therefore came
   out in a different order each run, which changed which tied documents
   survived the top-100 cutoff (15 tied pairs in the test run; headline
   metrics were unchanged to 4 decimals). Ties are now broken by doc id,
   and re-runs are byte-identical.
3. *The API silently dropped near-duplicates and returned fewer than k
   results.* Family members removed by `dedup_family` were never shown (the
   `same_family_as` field was always empty), and dedup ran after truncation
   to k. The API now over-fetches, dedups, fills k, and lists each
   representative's grouped family members.
4. *`alpha_sweep.csv`'s `map` column was actually MRR.* Renamed to `mrr`.

- **BM25 vs BM25F:** BM25 concatenates title+abstract+claims into one field;
  BM25F scores title/abstract/claims as separate BM25 indices combined with
  weights (claims weighted 2x, since they define legal scope). BM25F
  *underperforms* plain BM25 (NDCG@10 0.192 vs 0.211, p=0.042 unadjusted,
  p=0.205 after Holm correction, so suggestive rather than significant) —
  a real, reported finding, not a bug: the claims field is long
  and legally dense, and upweighting it appears to introduce more
  vocabulary noise than topical signal for this citation-based relevance
  proxy.
- **Semantic:** `BAAI/bge-small-en-v1.5` (384-dim), embedding title+abstract
  (claims omitted from the embedding input — too long relative to the
  model's practical context, and abstracts already summarize the invention).
  Swap-in path for a larger model: change `EMBEDDING_MODEL` in
  `src/index/vector_index.py` and re-run the index build.
- **Hybrid:** RRF (no tuning, `k=60`) and alpha-weighted linear combination
  of min-max-normalized BM25 and cosine scores, alpha swept on dev in steps
  of 0.1 (best: **0.5**, i.e. balanced) and applied to test only.
- **Reranking:** cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over
  the hybrid top-100 **hurts** performance here (see Section 4)
  — an off-the-shelf model trained on web/MS-MARCO passage relevance doesn't
  transfer to patent technical/legal text without domain adaptation. The
  LGBMRanker (LambdaMART) stretch does much better, but not on equal footing
  with everything else: its feature set includes a citation-popularity
  count, and our relevance labels *are* citations, so it has access to
  graph-structure signal no other method uses at all. See Section 4's
  discussion for why that makes it a genuinely different (not just
  "better-tuned") comparison.

## 4. Final Comparison Table

All methods scored on the same 62 test-split queries
(`data/qrels/qrels_test.txt`), generated by `src/eval/final_comparison.py`.
Two stage-to-stage differences reach p<0.05 unadjusted (Wilcoxon
signed-rank on paired per-query NDCG@10), and both are negative results:
BM25F underperforming BM25, and reranking underperforming hybrid. Because
five pairwise tests are reported together, the table also gives
Holm-Bonferroni-adjusted p-values; **after that correction neither
difference is significant** (both p=0.205). They should be read as
suggestive trends. The 62-query test split simply has little statistical
power for differences of 0.02–0.04 NDCG@10. Section 4.2 addresses this
with cross-validation over all 206 queries.

| Method | P@10 | R@10 | MRR | NDCG@10 | vs. prev (p) | vs. prev (Holm p) |
|---|---|---|---|---|---|---|
| BM25 | 0.0790 | 0.2534 | 0.2660 | 0.2113 | — | — |
| BM25F | 0.0742 | 0.2384 | 0.2458 | 0.1924 | 0.0420 | 0.2049 |
| Semantic | 0.0726 | 0.2312 | 0.2976 | 0.1990 | 0.8130 | 0.8130 |
| Hybrid (RRF) | 0.0919 | 0.2982 | 0.2792 | 0.2253 | 0.1546 | 0.4639 |
| Hybrid (alpha-weighted) | 0.0903 | 0.2884 | 0.3042 | 0.2357 | 0.3382 | 0.6764 |
| + Rerank (cross-encoder) | 0.0710 | 0.2268 | 0.2806 | 0.1944 | 0.0410 | 0.2049 |

**Cross-encoder reranking hurts** (p=0.041 unadjusted) — a cautionary
result: `ms-marco-MiniLM-L-6-v2` is trained on web-search passage relevance,
and that notion of "relevance" doesn't transfer to patent title/abstract
text matched against a citation-based ground truth without in-domain
fine-tuning. Worth stating plainly: bolting on a pretrained reranker is not
free lunch, and here it actively made the system worse than not reranking
at all — the best system in this table is **hybrid (alpha-weighted),
NDCG@10 = 0.2357**, not anything downstream of it.

**LTR (LambdaMART) stretch — a different comparison, not an apples-to-apples
one.** Trained on dev features (BM25, BM25F, cosine similarity, CPC-code
Jaccard overlap, filing-date proximity, leave-one-out citation-popularity
count), evaluated on test: **NDCG@10 = 0.2732, P@10 = 0.0968, MRR = 0.3588**
— above every other method. Before trusting that number, we checked it for
leakage: an earlier version scored NDCG@10 = 0.45 because the citation-count
feature included the very citation edge being predicted (a doc cited by
query Q had Q's own citation baked into its "popularity" score) — direct
label leakage. Fixed to a leave-one-out count (excluding the current
query's own edge) and the score dropped to a more credible 0.27, still
verified against the mean feature values: relevant docs average a
leave-one-out citation count of 3.45 vs. 0.14 for irrelevant ones. That gap
is real, not a bug — citation networks follow a preferential-attachment
pattern, so a handful of foundational patents get re-cited across many
different citing patents, and a popularity feature legitimately picks that
up. But **this is not a fair head-to-head with BM25/semantic/hybrid/
cross-encoder**: none of those methods see any citation-graph information
at all, while LTR does, and our relevance labels *are* citations. The
honest conclusion isn't "LambdaMART is a better reranking algorithm" — it's
"a citation-derived corpus rewards features derived from the same citation
graph," which is close to definitional. Reported as a distinct, caveated
result rather than folded into the main comparison table.

### 4.1 The temporal prior-art constraint

A document can only be prior art for a patent if it existed before that
patent was filed, but none of the retrieval methods above know this. On
average **3.4 of every method's top-10 results are patents filed after the
query patent**. They are topically similar, but they cannot be prior art for
it. `src/retrieve/temporal.py` drops them (docs filed on or before the
query's filing date are kept, including same-day filings), and
`src/eval/temporal_filter.py` re-scores every run with the filter applied:

| Method | Later-filed in top-10 | NDCG@10 | NDCG@10 filtered | Change | p (Holm) |
|---|---|---|---|---|---|
| BM25 | 3.55 | 0.2113 | 0.2223 | +5.2% | 0.061 |
| BM25F | 3.44 | 0.1924 | 0.2284 | +18.7% | 0.025 |
| Semantic | 3.37 | 0.1990 | 0.2292 | +15.2% | 0.025 |
| Hybrid (RRF) | 3.32 | 0.2253 | 0.2576 | +14.4% | 0.025 |
| Hybrid (alpha-weighted) | 3.44 | 0.2357 | **0.2631** | +11.6% | 0.025 |
| + Rerank (cross-encoder) | 3.37 | 0.1944 | 0.2232 | +14.8% | 0.025 |

The gain is significant for five of six methods even after Holm correction,
and the filtered hybrid is the best configuration in the project that uses
no citation-graph features. The gain isn't free: 11 of the 216 test qrels
(5.1%; 35 of 637 across all qrels) are citations to documents filed *after*
the citing patent, typically continuations or co-pending applications in
the same family. The filter makes those unreachable, so the reported gain is
net of that loss. Filing date is only an approximation of the legal test
(the reference's publication date vs. the query's priority date), and
priority dates are not in the corpus. The API applies this filter by
default in patent-as-query mode, and optionally, with a user-chosen
cutoff date, in free-text mode.

### 4.2 Cross-validated evaluation over all 206 queries

The fixed split leaves 62 test queries, and on those even the headline
comparison, hybrid vs. BM25 (0.2357 vs 0.2113), is not significant
(p=0.21). `src/eval/cross_validation.py` runs 5-fold cross-validation
instead: α is chosen on four folds and applied to the fifth, so every
query is scored exactly once as held-out and α is never selected on the
queries it is scored on. **α=0.5 was selected in all five folds**, so the
dev-split choice was not a fluke of that split.

| Method (n=206, CV) | P@10 | R@10 | MRR | NDCG@10 |
|---|---|---|---|---|
| BM25 | 0.0631 | 0.2238 | 0.2445 | 0.1790 |
| Semantic | 0.0626 | 0.2236 | 0.2397 | 0.1734 |
| Hybrid (CV α) | 0.0748 | 0.2724 | 0.2669 | 0.2067 |
| Hybrid (CV α) + prior-art filter | 0.0806 | 0.2909 | 0.2987 | **0.2320** |

| Comparison (NDCG@10) | p | p (Holm) |
|---|---|---|
| BM25 vs Hybrid | 0.0068 | 0.0068 |
| Semantic vs Hybrid | 0.0009 | 0.0018 |
| Hybrid vs Hybrid + prior-art filter | 3.3e-5 | 1.0e-4 |

With the extra power, **hybrid fusion significantly beats both of its
components** (+15.5% over BM25, +19.2% over semantic), and the temporal
filter adds a further significant +12.2%. Absolute numbers are lower than
on the test split because the full query set includes harder queries. The
comparison that matters is within this table, where every method is scored
on the same 206 queries.

## 5. Query Reduction Ablation

Patent-as-query topics, `patent_full` (title+abstract+claims+background,
uncut) vs reduced variants, scored with BM25 and hybrid on all 206 queries
(`results/query_reduction_ablation.csv`,
`results/query_reduction_ablation_simple.csv`). Changes are relative to
the full query with the same method.

| Query built from | Method | NDCG@10 | P@10 | R@10 | Change |
|---|---|---|---|---|---|
| Full patent | BM25 | 0.1933 | 0.0665 | 0.2390 | — |
| Full patent | Hybrid | 0.2123 | 0.0772 | 0.2793 | — |
| All claims + abstract† | BM25 | 0.1911 | 0.0660 | 0.2344 | −1.1% |
| All claims + abstract† | Hybrid | 0.1820 | 0.0680 | 0.2449 | −14.3% |
| TF-IDF top-40 terms | BM25 | 0.1916 | 0.0670 | 0.2405 | −0.9% |
| TF-IDF top-40 terms | Hybrid | 0.1892 | 0.0670 | 0.2398 | −10.9% |
| Claim 1 + abstract | BM25 | 0.1745 | 0.0597 | 0.2161 | **−9.7%** |
| Claim 1 + abstract | Hybrid | 0.1879 | 0.0675 | 0.2484 | −11.5% |

† This row was originally reported as "claim 1 + abstract". The claim
splitter never split HUPD's single-line claims field (Section 3), so it
actually measured *all* claims + abstract. It's kept, relabeled, because
it turns out to be an informative intermediate point. The true claim-1
variant was re-run after the fix.

**No reduction beats the full patent for either method.** The three
variants separate the two retrieval components cleanly:

- **BM25 cares about how much claim vocabulary survives.** Abstract plus
  all claims, or TF-IDF terms sampled from the whole document, lose only
  about 1% (within noise). Cutting down to claim 1, a median 14% of the
  claims text, costs 9.7%. Citation-relevant overlap is spread across the
  dependent claims, not concentrated in claim 1.
- **Hybrid loses 11–14% under every reduction.** The drop is largest for
  all-claims+abstract, even though that variant contains more text than
  claim-1+abstract. A likely explanation is input truncation: the
  embedding model reads at most 512 tokens, and that variant puts the long
  claims first, so the abstract is often cut off entirely. The full query
  starts with title + abstract, which is exactly what the corpus
  embeddings were built from. For the semantic side, what comes first in
  the query matters more than the total volume of text.

Practical consequence: the API's patent-as-query mode uses title +
abstract, the formulation every method was evaluated and α was tuned on.
A promising variant for future work, which avoids both failure modes, is a
title+abstract query for the embedding side combined with the full text
(or all claims) for the BM25 side.

*(Note: an earlier draft, run before the self-match fix in Section 3,
found a different ordering and explained it with a "coherent text embeds
better" theory. Neither that ordering nor the pre-fix claim-1 numbers
survived; the table above is the corrected, current state.)*

## 6. ANN Exact-vs-Approximate Tradeoff

Averaged over all 206 topics (`results/ann_tradeoff.csv`), IndexIVFFlat vs
exact IndexFlatIP search:

| | Exact (Flat) | Approximate (IVF) |
|---|---|---|
| Mean query latency | 2.13 ms | 0.29 ms |
| Recall@10 vs. exact | — (reference) | 96.0% |

IVF is **~7.4x faster** at **96% recall@10** relative to exact search — at
this corpus scale (14K docs) both are fast in absolute terms, but the
tradeoff curve is the point: nearly all of exact search's result quality is
recoverable at a fraction of the latency, which is the whole justification
for ANN indexing at larger scales where exact search stops being an option.

## 7. Result Diversification

Patent-family dedup (`src/retrieve/diversify.py`): TF-IDF cosine similarity
over title+abstract, collapsing results above a 0.85 threshold into their
highest-ranked representative. Applied as post-processing on the reranked
run (`results/runs/diversified.run`); not re-scored against qrels since
diversification trades relevance for coverage by design.

**Before/after example (query 10521475, "Travel-Related Cognitive
Profiles"):** the undiversified top-10 had 2 patent families dominating —
"Cognitive Profiles" (`10062031`/`9934328`, near-identical titles) and a
"Universal Cognitive Graph" cluster (`10860932`/`10860934`/`10860936`) —
together accounting for 4 of the top 5 slots. After dedup, those collapse
to one representative each, and the freed slots surface genuinely distinct
inventions that weren't in the original top-10 at all — including
"Method for Providing Travel Industry Optimized Cognitive Insights," which
given the query's own travel-specific framing arguably belongs higher than
some of what it replaced. Across all 206 topics, 45 queries had at least
one family collapsed, confirming this isn't a one-off case.

## 8. Qualitative Failure Analysis

Full annotated examples in `notebooks/error_analysis.ipynb` (10 examples,
5 per direction, executed with cached outputs). Summary:

**BM25 wins** when the relevant document shares specific, somewhat rare
technical terminology with the query that gives BM25 a strong exact-match
signal — semantic retrieval's top-3 in these cases is genuinely on-topic
but consistently misses the *specific* cited document, capturing "this is
about neuromorphic computing" without discriminating the finer-grained
mechanism (e.g. a shared "digital-to-analogue converter" detail) that
actually drove the citation.

**Semantic wins for three distinct, identifiable reasons**, not one vague
"embeddings are better at semantics":
1. **Vocabulary mismatch across specific-instance/general-category pairs**
   — e.g. query says "metal-air battery," relevant doc says "lithium-air
   battery" (lithium being a specific metal); BM25's exact terms don't
   bridge that, the embedding does.
2. **BM25's IDF weighting can backfire in a domain-narrow corpus** — a
   near-word-for-word-identical title pair scored zero BM25 credit because
   every shared term ("cathode," "lithium," "ion," "battery") is ubiquitous
   across this H01M-heavy corpus and gets IDF-down-weighted precisely
   because it's common *in this domain*, not because it's uninformative.
3. **A genuine, fixable tokenization bug.** "AIR-COOLED FUEL CELL VEHICLE"
   vs. "AIR-COOLED FUEL-CELL VEHICLE" — a near-exact title differing only
   by a hyphen — gets zero BM25 term overlap because our tokenizer
   (`src/index/bm25_index.py`) keeps internal hyphens attached to the
   token: `tokenize("FUEL CELL")` → `['fuel', 'cell']` but
   `tokenize("FUEL-CELL")` → `['fuel-cell']`, two disjoint token sets.
   Verified directly, not inferred. Not fixed in this project (flagged as
   scoped future work: strip hyphens before tokenizing, or emit both forms),
   but this is exactly the kind of concrete, actionable finding this
   analysis is meant to surface.

## 9. Scope Boundary

This is a **prior-art discovery tool**, not a patentability determination
system. It surfaces topically/citation-related patents for a human reviewer;
it does not assess novelty, non-obviousness, or any other legal
patentability criterion, and the passage-highlighting/summary layer is
explicitly grounded (extractive first, optional RAG-style generation
constrained to the retrieved passage) and labeled as such in the UI.

## 10. Limitations and Future Work

- **Statistical power.** The 62-query test split cannot resolve
  differences of a few NDCG points (Section 4). Prefer the cross-validated
  numbers (Section 4.2) for claims about which method is better, and apply
  a multiple-comparison correction whenever several tests are reported
  together.
- **Temporal cutoff uses filing dates**, not priority/publication dates
  (Section 4.1), and 5% of citation qrels point to later-filed documents.

- Corpus limited to 2 CPC classes and patents filed 2012–2018; a 3rd class
  would test whether fusion weights (alpha) transfer across domains.
- Citation-derived qrels (see Section 2 limitations).
- Stretch goals not required for the core pipeline: triplet-loss embedding
  fine-tuning on citation pairs (positives) vs. BM25 top-k non-cited
  (hard negatives); ColBERT-style late-interaction retrieval as a second
  semantic variant.
- **BM25 tokenizer doesn't normalize hyphenation** (Section 8): "fuel cell"
  and "fuel-cell" tokenize to disjoint token sets, costing BM25 exact
  matches on otherwise near-identical titles. Concrete, scoped fix not
  implemented here: strip hyphens before tokenizing, or emit both the
  hyphenated and de-hyphenated forms as tokens.
