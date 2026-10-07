# Rijksmuseum Graphic Arts Research Assistant — Documentation

This document explains what the prototype does, in what order, and where each piece of logic
lives. It follows the section numbers of `main.py` so you can jump between the
code and this document. The visual index is built separately, with the Colab notebook
`build_faiss_index_colab.ipynb` (Section 9).

---

## 1. What the system is

A hybrid retrieval-and-reasoning tool over a SQLite database of Rijksmuseum print and drawing
metadata. The database covers all periods (about 600,000 records); the tool and its visual index
are built for early modern prints and drawings (1450–1850).

Three design principles run through the whole prototype:

- **Traceability and grounding.** Every answer can be traced back to specific records, and the
  researcher can see exactly what the model read, what it didn't, and why.
- **The human stays in the loop.** The researcher evaluates the output; the tool's job is to give
  them the right instruments to do so (editable SQL, provenance per record, exact figures,
  warnings, exports).
- **Low resource, easy to upscale.** Small models run by default so the prototype is transferable;
  larger models can be swapped in by changing one line each (Section 10).

The components:

- **Text-to-SQL** turns a natural-language question into a SQL query against the metadata. This is
  the main engine and the most reliable signal: keyword matching against real catalogue fields.
  The database is opened **read-only**, so no generated query can change it.
- **Refinement** lets the LLM *edit* the previous SQL query ("only those after 1575") instead of
  writing a new one, so earlier conditions such as the creator are kept.
- **Vector search** (CLIP embeddings + FAISS) finds visually similar artworks. It always runs on an
  uploaded image (CLIP's strength: image-to-image similarity) and is **opt-in** for text queries,
  where it only ranks the SQL matches (or searches on its own if SQL finds nothing).
- **Reciprocal Rank Fusion (RRF)** merges SQL and vector results in image mode.
- **Three levels of results**, kept strictly apart:
  - the **result set**: all records that matched (table and CSV export);
  - the **profile**: exact facts about the result set, computed by pandas, never by the LLM;
  - the **sample**: the (max. 20) records the answer LLM reads in detail, chosen with a stated strategy.
- **A single LLM** (`Qwen2.5-1.5B-Instruct`, 4-bit on a GPU) writes the SQL, edits it for
  refinements and writes the final answer: one model, different system prompts.
- **A VLM** (`SmolVLM-500M-Instruct`) captions an uploaded image. The caption is shown to the
  researcher and given to the answer LLM as a soft, possibly inaccurate aid; it is *not* used as
  an SQL filter by default (Section 4.1).
- **Grounding written by the code, not the model.** Every answer gets a scope line (what it is
  based on), a sources list built from the database (inventory number, title, creator, date,
  persistent link) and visible warnings for invalid or missing citations.
- **Gradio** provides the UI: editable SQL, provenance per record, the profile, a full trace of
  each query, and exports of the results and the session log.

---

## 2. Startup sequence (what runs once, at launch)

Script sections 0–7 run **once**, top to bottom, before the Gradio UI is shown. If one of these
steps fails, the app doesn't start.

```
0. Read configuration (models, paths, column names, limits, thresholds)   (Section 0)
1. Connect to SQLite (read-only), read the schema, check column names     (Section 1)
2. Build the SQL, refine and synthesis prompts                            (Section 2)
3. Load the LLM                                                           (Section 3)
4. Load the VLM (skipped if ENABLE_VLM = False)                           (Section 4)
5. Load CLIP, load and VERIFY the FAISS index                             (Section 5)
7. Instantiate the ResearchEngine                                         (Section 7)
8. Launch the Gradio app                                                  (Section 8)
```

Notes on these steps:

- **A missing database or a wrong column name stops the app** with a clear message. A missing
  database raises `FileNotFoundError` (instead of SQLite silently creating an empty file); a wrong
  column name in section 0 raises an error that lists the real column names.
- **LLM quantization depends on hardware.** On a CUDA GPU the LLM loads in 4-bit (NF4); on CPU
  `bitsandbytes` can't quantize, so it loads unquantized (slower, more memory). If the VLM fails to
  load, the app still starts; image searches then run without a caption.
- **The FAISS index is only used if it provably belongs to this database and CLIP model.** The
  script checks the manifest written by the notebook: same CLIP model, same vector count and
  dimension, and the same *records fingerprint* (a SHA-256 over rowid and image URL of every
  eligible record). If anything differs, visual search is disabled with the reason, shown in the
  UI header and the trace, instead of silently returning wrong artworks. Text search keeps working.

**Main configuration values (section 0):**

| Setting | Value | What it controls |
|---|---|---|
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | SQL generation, refinement, answer synthesis |
| `VLM_MODEL_NAME` | `HuggingFaceTB/SmolVLM-500M-Instruct` | Image captioning |
| `CLIP_MODEL_NAME` | `sentence-transformers/clip-ViT-B-32` | Image/text embeddings; must match the index |
| `ENABLE_VLM` | `True` | Set to `False` on low-resource machines |
| `RETRIEVAL_TOP_K` | 20 | Max. records the answer LLM reads (the sample) |
| `VECTOR_SEARCH_K` | 15 | Nearest neighbours FAISS returns before thresholding |
| `MAX_RESULT_ROWS_DISPLAY` | 2000 | Rows shown in the "All matches" table (the export has all) |
| `VECTOR_SIMILARITY_THRESHOLD_TEXT` | 0.24 | Minimum similarity for text → image hits |
| `VECTOR_SIMILARITY_THRESHOLD_IMAGE` | 0.60 | Minimum similarity for image → image hits |
| `PLAUSIBLE_YEARS` | 1400–2030 | Dates outside this range are flagged as likely data errors |

**Column names (section 0):** `objectInventoryNumber`, `objectPersistentIdentifier`,
`objectTitle[1]`, `objectType[1]`, `objectCreator[1]`, `objectCreationDate[1]`, `objectImage`.
The answer LLM only reads title, object type, creator and creation date (`CONTEXT_FIELDS`), with
the inventory number as identifier.

**Where things live on disk:**

| Path | What it is |
|---|---|
| `DATABASE_PATH` (`../preprocessing/rma_artworks`) | The SQLite database (opened read-only) |
| `CURRENT_DIR/artworks.index` | FAISS vector index (built with the notebook) |
| `CURRENT_DIR/row_ids.npy` | Rowids matching the index vectors |
| `CURRENT_DIR/index_manifest.json` | How and from what the index was built; used for the startup check |
| `CURRENT_DIR/search_results.csv` | Written by "Export all matches (CSV)" |
| `CURRENT_DIR/session_log.csv` | Written by "Export session log (CSV)" |

---

## 3. The request types (what runs per user interaction)

After startup, everything happens inside `ResearchEngine`, triggered by the **Mode** selector
(New search, Refine current search, Follow-up) or the **Re-run edited SQL** button. New search,
refine and re-run all end in the same two methods: `retrieve()` (Section 3.5) and `_finish()`.

### 3.1 "New search" → `engine.new_search()`

```
Question and/or uploaded image
        │
        ▼
Image uploaded? ──yes──► analyze_image() (VLM) → caption, shown in the "VLM description" box
        │                  and kept for the whole retrieval session (follow-ups still see it)
        ▼
Question for SQL:
   - default: ONLY the typed text (the caption is NOT used)
   - experimental checkbox ON: typed text + two keywords from the caption (extract_filter_hints)
   - no typed text: SQL is skipped; an image-only search relies on CLIP
        │
        ▼
generate_sql() ──► LLM writes one SELECT query (stops at the first ';')
        ▼
run_sql() ──► safety checks, SELECT list rewritten to "SELECT rowid", executed read-only
        │      → sql_ids (IDs of all matches) + sql_for_ids (the same query, returning IDs only)
        ▼
CLIP:
   - image uploaded         → query vector from the IMAGE + global FAISS search (threshold 0.60)
   - text + checkbox ON     → query vector from the TEXT; a global search (threshold 0.24) only
                              if SQL found nothing, otherwise the vector only ranks the SQL matches
   - text + checkbox OFF    → skipped
   - index disabled         → skipped, the reason is written to the trace
        ▼
retrieve() ──► result set, sample, profile, context (Section 3.5)
        ▼
_finish() ──► conversation reset, answer (or "No artworks matched this query."), session log
```

### 3.2 "Refine current search" → `engine.refine_search()`

Used to narrow the current result set, e.g. "only those after 1575" after a search

```
Refinement instruction
        │
        ▼
No previous SQL? ──yes──► treated as a new search
        │
        ▼
refine_sql() ──► LLM edits the previous query (REFINE_SQL_SYS: keep ALL conditions,
        │         only add or adjust what the instruction asks for)
        ▼
run_sql() ──► new sql_ids
        ▼
Subset check: are all new matches inside the previous result set?
   - yes → trace: "All N matches lie within the previous M"
   - no  → visible NOTE at the top of the answer: the refinement widened or changed the search
        ▼
retrieve() with the vector hits and query vector of the last search (no new CLIP run)
        ▼
_finish() with the chained question, e.g.
"Which prints were produced by Cornelis Cort? (refined: Only give those after 1575)"
```

Refinements can be stacked; the edited SQL appears in the SQL box.

### 3.3 "Follow-up" → `engine.follow_up()`

Reasons over the records **already in the context**: no SQL, no CLIP, no new retrieval. Use it for
questions about the sample ("compare #2 and #3"), not to narrow the full result set (that's what
Refine is for). If there is no context yet, it falls back to a new search. The answer is added to
the conversation history; the tables stay as they were.

### 3.4 "Re-run edited SQL" → `engine.rerun_edited_sql()`

The researcher edits the SQL box directly. The query goes through the same safety checks; a
rejected query shows "SQL rejected: …" and leaves the current results untouched. Otherwise it runs
through `retrieve()` with the vector half of the last search, and the answer is rewritten
("Describe and compare the retrieved artworks.").

### 3.5 What `retrieve()` does

SQL returns rows in rowid order, which says nothing about relevance. `retrieve()` decides which
records the LLM reads and prepares everything else:

```
1. Text mode with SQL matches: vector hits are ignored (CLIP ranks, it doesn't add records)
2. Query vector + SQL matches: score_candidates() ranks the SQL matches by CLIP similarity
   (matches without an indexed image go to the end)
3. load_records() ──► the FULL result set as one DataFrame
                      (full records for all SQL matches (fetched by running sql_for_ids inside a second query plus the vector hits))
4. The sample (max. RETRIEVAL_TOP_K records) and its ranking note:
   - image mode           : RRF of the ranked SQL matches and CLIP's global hits
   - SQL found nothing    : CLIP's global hits
   - text + query vector  : the SQL matches most similar to the question
   - text, no vector      : chronological_sample(): evenly spread over ALL matches by date
5. Columns added per record: "Artwork #" (position in the sample, empty if not in it),
   "found_via" (SQL / Vector / SQL+Vector) and "vector_similarity"
6. build_profile() over ALL matches
7. Context text for the LLM, one line per artwork:
   [Artwork #3 | RP-P-OB-52.587] title: … | object type: … | creator: … | creation date: …
   and captioned gallery images ("Artwork #3 (RP-P-OB-52.587) - title")
```

The chronological sample takes the middle record of each of 20 equal slices of the date-sorted
matches, so it mirrors the date distribution of the full set.

---

## 4. The prompts and generation settings

| Prompt | Used by | Purpose |
|---|---|---|
| `SQL_FILTER_SYS` | `generate_sql()` | Question → one SQL query: the shared rules (`SQL_RULES`) + three few-shot examples |
| `REFINE_SQL_SYS` | `refine_sql()` | Previous query + instruction → edited query: same rules + two examples |
| `SYNTHESIS_SYS_TEMPLATE` | `synthesize()` | Answer over the context, with a fixed structure for new questions |

The shared SQL rules: the database covers all periods, but queries are always restricted to
1450–1850 (or a narrower period the question gives) unless the researcher explicitly asks for other
years; exact bracketed column names; Dutch search terms; subjects via several Dutch title words
combined with OR; creator names matched as separate 'Lastname' and 'Firstname' parts.

The synthesis prompt asks, for a **new** question, for three parts: **Answer** (figures about the
whole result set only from the profile), **Observations** (each starting with "Artwork #N (year)")
and **Limits**. Follow-ups are answered directly but must still cite. The code tells the model
explicitly whether it's handling a new question or a follow-up.

At run time the synthesis prompt also receives the profile; a note on how many artworks matched
versus how many are in the context and how they were chosen (or that the context is complete);
and, after an image upload, the VLM caption labelled as a rough, possibly inaccurate aid.

**Generation settings** (all calls go through `generate_chat()`):

| Call | Decoding | repetition_penalty | max_new_tokens | Other |
|---|---|---|---|---|
| `generate_sql()` | greedy | 1.0 | 250 | stops at the first `;` |
| `refine_sql()` | greedy | 1.0 | 300 | stops at the first `;` |
| `extract_filter_hints()` | greedy | 1.0 | 15 | — |
| `synthesize()` | temperature 0.3 | 1.05 | 900 | top_p 0.9, top_k 50 |

`repetition_penalty` also penalizes every token already in the *prompt*, so the SQL calls use 1.0:
they must copy column names from the schema. Looping is prevented by stopping at `;` instead.

### 4.1 What each component does for an uploaded image

| Component | Role | Influence |
|---|---|---|
| **CLIP + FAISS** | Finds artworks that *look like* the image; ranks the SQL matches by similarity to it | Ranked list in the fusion (threshold 0.60) |
| **SQL** | Handles what the researcher *typed* | Ranked list in the fusion; doesn't filter the CLIP results |
| **VLM** | Describes the image in words | Soft only: shown to the researcher, given to the answer LLM as a rough aid |

**Why the caption is not an SQL filter by default:** a VLM sometimes names the wrong medium or
period. As context, the answer LLM can weigh that; as SQL, it becomes a hard condition that
silently excludes the right artworks. The experimental checkbox adds only two condensed keywords
(object type, subject), and the trace shows them.

---

## 5. Data flow

```
                     ┌──► run_sql() ──► sql_ids + sql_for_ids ──┐
Question ──► LLM ────┤                                           │
                     └── (refine: edits the previous SQL)        ▼
                                                             retrieve()
Image ──► CLIP ──► query vector ──► FAISS global search ─────►  │  ranking / RRF
  │                      └────────► score_candidates() ──────►  │  load_records() → ALL matches
  │                                                              │  sample + ranking note
  └──► VLM caption ───────────────────────────────┐              │  profile
                                                  ▼              ▼
                                    synthesize() ◄── context (sample) + profile + notes
                                         │
                                         ▼
                   attach_sources(): scope line, sources from the database, warnings
                                         │
UI: answer · profile · gallery · "Records in the model's context" · "All matches" · trace · exports
```

For an uploaded image the components are cleanly separated: CLIP drives retrieval by visual
similarity, SQL by what you typed, and the VLM caption informs the *answer*, not the retrieval.
For text queries CLIP is off by default; when you opt in, it only ranks what SQL found.

---

## 6. Traceability: what you can inspect, and where

| Where | What it shows | When to use it |
|---|---|---|
| **Answer box** | Scope line, answer, sources list from the database, warnings | Checking every cited record against the collection (persistent link) |
| **Result set profile** | Count, distinct titles, date range and distribution, flagged dates, object types, top creators, image and index coverage | Any statement about the whole result set; spotting data problems |
| **Records in the model's context** | The sample: `Artwork #`, `found_via`, `vector_similarity`, all fields | What exactly the answer is based on |
| **All matches** | The full result set, chronological, with the same three columns | What the model did *not* read |
| **Gallery** | Full-resolution images captioned "Artwork #N (inventory number) - title" | Matching images to citations |
| **VLM description** | The literal caption of an uploaded image | Checking whether an odd answer traces back to a bad caption |
| **Retrieval trace** | Every step: SQL, matches, CLIP, ranking, sample, profile, citation check | Explaining a specific answer, also via a shared `gradio.live` link |
| **Session log / CSV** | Per answer: query, SQL, answer, total matches, how the sample was chosen, the inventory numbers read, the LLM | Auditing or reporting a session |
| **search_results.csv** | All matches of the current search | Further analysis outside the tool |

**Trace lines worth knowing:**

- `[Rank] Sample: …` and `[Rank] Final selection (20 of N …)` — how the sample was chosen and how
  much of the result set it covers.
- `[Refine] …` — the previous and edited SQL, and whether the refinement really narrowed the set.
- `[Vector] Visual search disabled: …` — why CLIP didn't run.
- `[Check] …` — the citation check.

**Reading `vector_similarity`:** a row marked `SQL` can have a similarity score; it comes from
ranking the SQL matches, not from CLIP finding it. Only `Vector` and `SQL+Vector` rows were found
by CLIP's global search. A score of 0.61, just above the 0.60 threshold, is a very different
situation from 0.95, even though both rows are labelled "Vector"!

---

## 7. Engine state reference

**Retrieval state** — overwritten by new search, refine and re-run; untouched by follow-ups:
- `results` — ALL matches (DataFrame): the "All matches" table, the profile and the export
- `context_df` — the sample the LLM reads, ordered by `Artwork #`
- `context_str` — the context text given to the LLM
- `images` — (url, caption) pairs for the gallery
- `profile` — exact facts about `results`
- `ranking_note` — how the sample was chosen; told to the LLM and logged
- `image_analysis` — the VLM caption, kept for follow-ups and re-runs

**Kept for refinements and re-runs:**
- `last_sql`, `last_sql_ids` — the last query and its matches (for the subset check)
- `last_question` — the question the current results answer, including refinements
- `last_vector_hits`, `last_query_vec`, `last_search_was_image` — the vector half of the last
  search, so refinements and edited SQL don't re-run CLIP

**Fixed lookup** (built once): `rowid_to_pos` — rowid → position in the FAISS index.

**Conversation state:** `history` — reset by every retrieval, extended by every answer.

**Trace:** `_trace_buffer` — reset at the start of every request.

**Session log:** `session_log` — never resets; exportable as CSV.

---

## 8. Known limitations

**Answer quality (mainly model size)**

- **The 1.5B LLM invents interpretations.** It sees only title, type, creator and date, yet it
  "explains" what prints depict and mistranslates Dutch titles (e.g. "Hercules voert Diomedes aan
  zijn paarden" as Hercules *leading* Diomedes). The answers are fluent and confident, so they need
  checking. It also often ignores the answer structure and cites no records; the citation warning
  and the code-written scope line catch this, but they can't correct the content. See Section 10.
- **The citation check is shallow.** It confirms that cited artwork numbers exist in the context,
  not that what the answer says about them is correct.
- **The VLM caption is only as good as the model.** Compare it with the image itself.
- **Answers are sampled (temperature 0.3),** so the same question can give slightly different
  answers. The session log records what was actually answered.

**Retrieval and scope**

- **Only the sample (max. 20 records) is read in detail.** Statements about the whole set must come
  from the profile; follow-ups only concern the sample. Narrowing the full set needs Refine.
- **In image mode, typed text boosts but doesn't restrict the results.** SQL and CLIP are fused as
  two ranked lists, so visually similar works by other artists can appear. Check `found_via`.
- **Image-only searches have no metadata filtering by default.** Results can look alike but differ
  in medium, period or object type.
- **CLIP nearest neighbours are not attribution.** For "who made this?", the creator of the closest
  image is only as trustworthy as its `vector_similarity`.
- **CLIP center-crops images.** For tall or wide prints the edges don't count in the similarity;
  this applies equally to the index and to uploaded images.
- **The similarity thresholds (0.24 text, 0.60 image) are starting points,** not validated cutoffs.
- **Generated SQL can't count or group.** The SELECT list is always rewritten to `rowid`, so
  `COUNT`/`GROUP BY` questions rely on the profile or the exported CSV.

**Data**

- **Creation dates are compared as text** in the SQL (`BETWEEN '1450' AND '1850'`). This works for
  clean four-digit years; a handful of typo dates (e.g. '180', '18885') behave oddly. The profile
  lists them so they can be fixed in preprocessing.
- **One creator per record.** Prints are collaborative objects (designer, engraver, publisher); the
  database records one name, so some works appear under the designer rather than the engraver.
- **Records with the same title** can be several impressions, states or parts of one print; the
  profile reports distinct titles next to the total.
- **Dates after the artist's death** (e.g. Cort, d. 1578) usually mean later impressions or
  editions; the creation date is not always "when the artist made it".

**Technical**

- **No KV-cache reuse across turns:** long follow-up conversations get slower.
- **`main_table = tables[0]`** assumes a single-table database.
- **One shared engine:** fine for one researcher; several simultaneous users of a shared link would
  share results and conversation (per-user state would need `gr.State`).
- **Without a GPU the LLM runs unquantized:** slow, and more memory.

---

## 9. Building the visual index (Colab notebook)

`build_faiss_index_colab.ipynb` builds `artworks.index`, `row_ids.npy` and `index_manifest.json` on
a Colab T4 GPU.

1. Upload the database once to Google Drive: `MyDrive/rma_index/rma_artworks`.
2. Run the cells in order. The notebook copies the database to the Colab disk, selects all eligible
   records (print or drawing, 1450–1850, with an image URL) and computes the records fingerprint.
3. Images are downloaded in parallel (`=s400` versions; CLIP only uses 224 px), with retries and
   backoff for temporary errors, and embedded with CLIP in FP16. The aspect ratio is kept; CLIP's
   own preprocessing resizes and crops, exactly as for a query image in the app.
4. Every chunk of 1,024 records is saved as a shard on Drive. After a disconnect, re-run the cells:
   the build resumes at the first missing chunk. Shards from another database or model are refused.
5. The optional retry cell tries failed downloads again, more gently.
6. The final cell builds the index and manifest and writes `failed_records.csv` (records without a
   vector, and why). The last cell re-embeds random images as a sanity check (expect ~0.99+).
7. Download the three files from Drive into the folder of the assistant script.

---

## 10. Scaling up

The biggest remaining improvement is the size of the models; the retrieval, grounding and
traceability layers don't depend on it. Each model is one line in section 0.

| Component | Default (low resource) | Upscale option | Notes |
|---|---|---|---|
| LLM | Qwen2.5-1.5B-Instruct | Qwen2.5-3B / 7B-Instruct (4-bit) | 7B needs roughly 5–6 GB VRAM; much better Dutch, structure and citations. Raise `RETRIEVAL_TOP_K` with it |
| VLM | SmolVLM-500M-Instruct | e.g. Qwen2.5-VL | Better medium and technique descriptions |
| CLIP | clip-ViT-B-32 | clip-ViT-L-14 | Finer detail (hatching, states); needs a rebuilt index (`BATCH_SIZE = 64` in the notebook) and the same name in the app |