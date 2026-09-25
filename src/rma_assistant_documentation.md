# Rijksmuseum Graphic Arts Research Assistant — Documentation

This document explains what the prototype (`rma_assistant_v2.py`) does, in what order, and
where each piece of logic lives. It's organized around the script's own section numbers so
you can jump between the code and this doc easily.

---

## 1. What the system is

A hybrid retrieval-and-reasoning tool over a SQLite database of Rijksmuseum print/drawing
metadata:

- **Text-to-SQL** turns a natural-language question into a SQL query against the metadata.
  This is the main engine — it does exact-ish keyword matching against real catalog fields
  and is the most reliable signal in the pipeline.
- **Vector search** (CLIP embeddings + FAISS) finds visually similar artworks. It always runs
  on an uploaded image (CLIP's genuine strength: image-to-image similarity) but is **opt-in**
  for text-only queries, since CLIP's text-image alignment is only reliable for short,
  caption-like, topic-heavy phrasing ("apes", "landscapes"), not specific or technical
  questions. A similarity threshold also drops weak matches before they can reach fusion.
- **Reciprocal Rank Fusion (RRF)** merges the SQL and vector result lists into one ranked set.
- **A single LLM** (`Qwen2.5-7B-Instruct`, 4-bit quantized) does both the SQL generation and
  the final answer synthesis — one model, two different system prompts.
- **A VLM** (SmolVLM-500M) captions an uploaded image, feeding that description into both SQL
  generation and synthesis.
- **Gradio** provides the UI, including a way to inspect and edit the generated SQL, see why
  each retrieved row was found (SQL, vector, or both, with similarity scores), read a full
  step-by-step trace of a query's execution, and hold a follow-up conversation without
  re-running retrieval.

---

## 2. Startup sequence (what runs once, at launch)

Everything in script sections 1–7 runs **once**, top to bottom, when the script starts —
before the Gradio UI is even shown. This is important: if any of these steps fail (bad
database path, missing model), the app won't launch at all.

```
1. Connect to SQLite database, read schema        (Section 1)
2. Build SQL_FILTER_SYS and SYNTHESIS_SYS prompts   (Section 2)
3. Load the LLM (Qwen2.5-7B, 4-bit quantized)       (Section 3)
4. Load the VLM (SmolVLM-500M)                      (Section 4)
5. Load the CLIP embedding model                    (Section 5)
6. Build or load the FAISS image index              (Section 5)
7. Instantiate the ResearchEngine                   (Section 7)
8. Launch the Gradio app                             (Section 8)
```

Steps 3–6 are the slow ones (model downloads/loading, and — on first run only — downloading
and embedding up to `MAX_INDEX_IMAGES` images). Once `artworks.index` and `row_ids.npy` exist
on disk, step 6 just loads them instead of rebuilding, so subsequent runs are much faster.

**Where things live on disk:**

| Path | What it is |
|---|---|
| `DATABASE_PATH` (`../preprocessing/rma_artworks`) | The SQLite database (input, read-only) |
| `CURRENT_DIR/artworks.index` | Cached FAISS vector index (built on first run) |
| `CURRENT_DIR/row_ids.npy` | Row IDs matching the FAISS index vectors |
| `CURRENT_DIR/session_log.csv` | Written when you click "Export session log" |

---

## 3. The three request types (what runs per user interaction)

After startup, everything else happens inside `ResearchEngine`, triggered by one of three UI
actions. This is the core logic to understand.

### 3.1 "New search" → `engine.new_search()`

Used when: mode toggle is set to "New search" and you click Submit.

```
User types a question and/or uploads an image
        │
        ▼
Image uploaded? ──yes──► analyze_image() runs the VLM (Section 4)
        │                 → produces a text description (medium, composition, etc.)
        │                 → printed to console AND returned for the UI's "VLM description" box
        no
        │
        ▼
Combine: user's text question + (VLM description, if any)
        │
        ▼
generate_sql() ──► LLM writes a SQL query (Section 3, SQL_FILTER_SYS prompt)
        │
        ▼
run_sql() ──► query executed against SQLite → list of rowids (sql_ids)
        │
        ▼
Vector search gate:
   - image uploaded?              → CLIP always searches (its real strength)
   - text only + checkbox OFF     → vector search SKIPPED entirely
   - text only + checkbox ON      → CLIP searches the text (opt-in, since this
                                      only works well for short, topic-heavy
                                      phrasing like "apes" or "landscapes")
        │
        ▼ (if running)
vector_search() ──► CLIP-encodes the text and/or image, searches FAISS,
        │            DROPS any hit below VECTOR_SIMILARITY_THRESHOLD (0.24),
        │            returns (rowid, similarity_score) pairs (vector_ids)
        ▼
reciprocal_rank_fusion(sql_ids, vector_ids) ──► merged, ranked rowids (hybrid_ids)
        │
        ▼
source_map built: for each hybrid_id, records whether it came from SQL,
        │           vector search, or both, plus its similarity score
        ▼
fetch_context(hybrid_ids, source_map) ──► pulls full rows from SQLite,
        │                      builds (a) a text context block for the LLM,
        │                      (b) image URLs for the gallery,
        │                      (c) a DataFrame for the results table, now
        │                          including "found_via" and "vector_similarity"
        │                          columns so provenance is visible per row
        ▼
Engine STATE IS RESET:
   current_context_str, current_hybrid_ids, current_image_urls,
   current_source_map updated
   self.history = []   (conversation memory cleared — this is a fresh topic)
        │
        ▼
synthesize() ──► LLM writes the answer (Section 7, SYNTHESIS_SYS_TEMPLATE prompt)
        │         using the fresh context, no prior conversation
        ▼
Every step above is also appended to a trace buffer (printed to console AND
returned to the UI's "Retrieval trace" accordion), logged to session_log,
and returned to the UI
```

**Key point:** a new search always overwrites the retrieval state *and* clears conversation
history. This is deliberate — asking a brand-new question shouldn't drag old context into the
LLM's reasoning. It also always resets the trace buffer, so the accordion always shows the
steps for the *current* query, not an accumulation across turns.

### 3.2 "Follow-up" → `engine.follow_up()`

Used when: mode toggle is set to "Follow-up" and you click Submit.

```
User types a follow-up question (no image, no retrieval)
        │
        ▼
current_context_str empty? ──yes──► falls back to new_search() automatically
        │
        no
        ▼
Trace records: "no retrieval re-run", how many artworks are being reused,
        │        and how many prior conversation turns exist
        ▼
synthesize() ──► LLM answers using:
        │          - the SAME current_context_str as the last search
        │          - self.history (all prior Q&A turns this session)
        │          - the new question appended
        ▼
self.history grows by one more (question, answer) pair
        │
        ▼
Results table is rebuilt from the SAME hybrid_ids and source_map as the last
search, so "found_via" / "vector_similarity" columns stay consistent
        │
        ▼
Logged to session_log, trace returned to the UI, returned to the UI
```

**Key point:** no SQL, no vector search, no FAISS lookup happens here at all. It's purely the
LLM reasoning over data it already retrieved, plus the running conversation. This is what
makes multi-turn questions ("compare #2 and #3", "what about the same period in drawings?")
possible and fast.

### 3.3 "Re-run edited SQL" → `engine.rerun_edited_sql()`

Used when: you edit the text in the SQL box and click "Re-run edited SQL".

```
Researcher edits the SQL box directly and clicks "Re-run edited SQL"
        │
        ▼
run_sql(edited_sql) ──► safety checks (SELECT-only, no dangerous keywords),
        │                 executed against SQLite → new sql_ids
        ▼
reciprocal_rank_fusion(sql_ids, last_vector_ids) ──► reuses the vector
        │            results AND similarity scores from the last search
        │            (no new CLIP/FAISS work — last_vector_scores is reused)
        ▼
source_map rebuilt from the new sql_ids + the reused vector scores, so the
        │           results table's "found_via"/"vector_similarity" columns
        │           correctly reflect the edited query
        ▼
fetch_context() ──► rebuilds context, image list, table from the new hybrid_ids
        │
        ▼
Engine state updated: current_context_str, current_hybrid_ids,
current_source_map, last_sql_query
self.history = []   (treated as a new retrieval, so conversation resets)
        │
        ▼
synthesize() ──► LLM re-answers with the new context
        │
        ▼
Logged to session_log, trace returned to the UI, returned to the UI
```

**Key point:** this is how you correct or override the LLM's SQL guess without going through
natural language again — you're directly controlling retrieval, and the vector-search half of
the fusion is preserved from the original search rather than being thrown away.

---

## 4. The two prompts and why they're different

| Prompt | Used by | Purpose | Style |
|---|---|---|---|
| `SQL_FILTER_SYS` | `generate_sql()` | Translate NL question (+ image description) into one SQL query | Rigid: few-shot examples, exact column names, Dutch-term translation rules, "return ONLY the raw SQL" |
| `SYNTHESIS_SYS_TEMPLATE` | `synthesize()` | Reason over retrieved artworks and answer the researcher | Open: asks for comparison, patterns, and flagged uncertainty rather than a forced per-item list |

Both go through the same `generate_chat()` function and the same underlying model — only the
system prompt and generation settings (temperature, `max_new_tokens`) differ per call. SQL
generation uses `temperature=0.0` (deterministic); synthesis uses `temperature=0.6` (more
natural, varied phrasing).

---

## 5. Data that flows between components

```
SQLite DB  ──rows──►  fetch_context()  ──text block──►  LLM (synthesis)
    │                        │
    │                        ├──image URLs──► Gradio Gallery
    │                        ├──DataFrame────► Gradio results table
    │                        │                   (with found_via / vector_similarity
    │                        │                    columns from source_map)
    │
    └──rowids (via SQL)──┐
                          ├──► reciprocal_rank_fusion() ──► hybrid_ids ──► source_map
    CLIP+FAISS            │         (only reached if the vector-search gate, below,
    (gated + thresholded) │          let vector_ids through in the first place)
    └──rowids+scores (via vector search, if gate open)──┘

Vector-search gate (before the fusion step above):
    image uploaded?              → CLIP always searches (its real strength)
    text only + checkbox unchecked → SKIPPED, vector_ids = []
    text only + checkbox checked   → CLIP searches the text (opt-in)
    Every hit, regardless of path, is dropped if similarity < VECTOR_SIMILARITY_THRESHOLD

Uploaded image ──► VLM (analyze_image) ──text description──┬──► fed into SQL generation
                                                              ├──► fed into synthesis prompt
                                                              └──► shown in UI's "VLM description" box

Every step above ──► ResearchEngine._trace() ──► printed to console AND
                                                   shown in the UI's "Retrieval trace" accordion
```

The important thing to notice: an uploaded image influences the answer through **two
independent paths** — its CLIP embedding drives vector search, and its VLM-generated text
description drives both the SQL query and the synthesis prompt. This is what lets an image
upload trigger metadata filtering (e.g. period, object type), not just visual similarity. A
third thing worth noticing: for **text-only** queries, the CLIP path is off by default —
vector search only contributes when you're searching by image, or when you've explicitly
opted a text query into it.

---

## 6. Traceability: what you can inspect, and where

Every retrieval-driving action is now logged in three places, at different levels of detail,
so you can pick the right one for what you're checking:

| Where | What it shows | When to use it |
|---|---|---|
| **Console / Colab cell output** | Raw `print()` statements as each step runs (VLM caption, generated SQL, matched rowids, vector candidates + scores, RRF inputs/outputs) | Live debugging while the script runs in the notebook |
| **UI → "Retrieval trace" accordion** | The exact same step-by-step log, but returned with the answer and shown in the Gradio app itself | Explaining a specific answer to a colleague using only the shared `gradio.live` link — they won't see the Colab console at all |
| **UI → "Retrieved records" table** | Per-artwork `found_via` (SQL / Vector / SQL+Vector) and `vector_similarity` columns | Quickly judging, row by row, why a given artwork made it into the answer |
| **UI → "VLM description" box** | The literal caption the VLM generated for an uploaded image | Checking whether a strange SQL query or a wrong-seeming answer traces back to a bad VLM read of the image |
| **Session log / exported CSV** | Every query, generated SQL, and answer for the whole session (not the step-by-step trace) | Auditing a full session afterward, or sharing what was asked and answered |

**Why this matters for the earlier CLIP-attribution concern:** if you upload a print and ask
"who made this," the trace will show you explicitly whether that answer came from a
vector-search hit (and at what similarity score) or from a SQL match — so you no longer have
to take the answer's confidence at face value. A `vector_similarity` of 0.26 sitting just
above the 0.24 threshold is a very different situation from 0.9, even though both would be
labeled "Vector" in the table.

---

## 7. Engine state reference

`ResearchEngine` holds three kinds of state, and knowing which is which explains most of the
control flow above:

**Retrieval state** (what was found) — overwritten by `new_search()` and `rerun_edited_sql()`,
left untouched by `follow_up()`:
- `current_context_str` — the text block fed to the LLM
- `current_hybrid_ids` — the ranked rowids currently "in view"
- `current_image_urls` — images shown in the gallery
- `current_source_map` — per-rowid dict of `{sql: bool, vector: bool, vector_score: float}`,
  used to rebuild the results table's provenance columns on every turn, including follow-ups
- `last_sql_query`, `last_vector_ids`, `last_vector_scores` — kept so edited-SQL re-runs can
  reuse the vector half (and its scores) without re-running CLIP/FAISS

**Conversation state** (what's been discussed) — reset to `[]` by `new_search()` and
`rerun_edited_sql()`, appended to by `follow_up()` and by the first answer of a new search:
- `self.history` — list of `{"role", "content"}` turns passed to the LLM on every synthesis
  call, so it can refer back to what it already said

**Trace state** (how the last query was resolved) — reset at the start of every
`new_search()`, `follow_up()`, and `rerun_edited_sql()` call:
- `self._trace_buffer` / `self.last_trace` — the step-by-step log described in Section 6,
  rebuilt fresh each time so it never mixes steps from different queries

**Session log** (never reset, grows for the life of the process):
- `self.session_log` — every query/SQL/answer across all three request types, exportable to
  `session_log.csv` via the UI button

---

## 8. Known limitations to keep in mind

- **No KV-cache reuse across turns.** Each `generate_chat()` call reprocesses the full prompt
  (context + history) from scratch. Long follow-up conversations will get slower as
  `self.history` grows, since there's no persistent cache between calls.
- **`main_table = tables[0]`** assumes the first table SQLite returns is the right one. Fine
  for a single-table database; revisit if you add related tables.
- **The dangerous-keyword check in `run_sql()`** is a simple substring filter, not a full SQL
  parser — a reasonable guard for a research tool, not a hardened security boundary.
- **FAISS index build is synchronous and runs at startup** if no cached index exists;
  expect a wait on first launch proportional to `MAX_INDEX_IMAGES`.
- **`VECTOR_SIMILARITY_THRESHOLD` (0.24) is a starting point, not a validated cutoff.** It was
  chosen as a reasonable floor for CLIP ViT-B/32 cosine similarity, but the right value
  depends on your actual image collection's score distribution — worth checking against a
  few known-good and known-irrelevant matches before trusting it fully.
- **The "include visual similarity for text" checkbox defaults to off**, so a plain topic-heavy
  query like "farm animals" will get *no* vector-search contribution unless a researcher
  remembers to tick it. This trades a small amount of recall on that specific query style for
  not diluting the far more common, SQL-friendly query with an unreliable signal.
- **CLIP nearest-neighbor hits are not attribution.** Uploading a print and asking "who made
  this" still just reports whichever visually closest indexed image's creator field happens to
  say — the `vector_similarity` score in the results table is the only signal for how much to
  trust that, and nothing currently blocks a low-confidence match from still producing a
  fluent, confident-sounding answer.
