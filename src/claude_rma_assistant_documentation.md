# Rijksmuseum Graphic Arts Research Assistant — Technical Documentation

This document explains what the prototype does, in what order, and where each piece of logic lives. It is
organised around the script's own section numbers (`main.py`, sections 0–8), so you can jump
between the code and this document easily.

*Version 1.0 of the prototype, 21 October 2026. Describes the script with the annotation layer (researcher insights, system proposals, SQL-searchable annotations) and the SQL guardrail. Section 12 lists what changed since the previous version of this document.*

---

## 1. Design principles

Every feature is tested against four principles:

- **Grounded.** Every contribution of the system points to records, scores or annotations. Facts about the
  data (counts, sources, which annotations apply) are written by code, never by the language model.
- **Human authority.** Nothing becomes knowledge without the researcher's confirmation. System proposals
  wait until the researcher confirms or rejects them; LLM drafts are never saved without a click.
- **Provenance.** Every piece of knowledge records who proposed it (researcher or system), who confirmed
  it, when, and on what grounds.
- **Initiative, small and dismissible.** The system may start a thread (proposals), but the researcher can
  leave, reject or switch it off (propagation grounds per insight).
- **No case knowledge in the code.** What is known about a particular maker, publisher or print belongs in
  the annotations, where it is visible and has provenance, not in configuration or prompts. The code is
  specific to this collection (its columns, scope, Dutch vocabulary and examples), but never to a case.

The underlying idea: the prototype started as deterministic, traceable retrieval. The annotation layer adds
a second direction, from the researcher into the system, so knowledge accumulates across sessions and the
system can extend it.

---

## 2. What the system is

A hybrid retrieval-and-reasoning tool over a SQLite database of Rijksmuseum print and drawing metadata
(inventory number, persistent identifier, title, object type, creator, creation date, image URL), with a
writable layer of researcher knowledge on top.

| Component | Role |
|---|---|
| **Text-to-SQL** (LLM) | Turns a question into one SQL query over the metadata and the confirmed annotations. The main retrieval engine. The museum database is opened **read-only** |
| **CLIP + FAISS** | Visual similarity. Always used for an uploaded image; opt-in for text questions, where it only ranks the SQL matches. Also finds "same design" candidates for propagation |
| **Answer LLM** | Reads a sample of at most 20 records (plus their annotations) and writes the answer. Same model as text-to-SQL, different prompt |
| **VLM** | Describes an uploaded image in words. Shown to the researcher and given to the answer LLM as a rough aid; never a search filter by default |
| **Annotation layer** | Researcher insights stored in `annotations.db`, with provenance. Searchable by SQL, shown in the tables, given to the answer LLM |
| **Propagation** | After an insight is saved, the system proposes other prints it may apply to, on grounds the researcher chooses |
| **Drafting** (LLM) | Turns the researcher's own words into a draft annotation (artworks, label, note) for the researcher to check and save |
| **Checks** (code) | Scope line, sources, citation warnings, "Known about these records" block, trace and session log |
| **Gradio UI** | Search, refine, follow-up, editable SQL, tables, gallery, annotation panel, proposals, trace, exports |

**Models (section 0):**

| Setting | Default | Notes |
|---|---|---|
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | Fine for quick interface tests on modest hardware. **All tests were run with `Qwen/Qwen2.5-7B-Instruct`**: set this for real use and for the frozen setup (section 11) |
| `VLM_MODEL_NAME` | `HuggingFaceTB/SmolVLM-500M-Instruct` | Captioning of uploaded images only. `ENABLE_VLM = False` disables it |
| `CLIP_MODEL_NAME` | `sentence-transformers/clip-ViT-B-32` | Must be the model the FAISS index was built with |

---

## 3. Startup sequence (what runs once, at launch)

Sections 0–7 run once, top to bottom, before the interface opens. If a step fails (wrong database path,
wrong column name, missing model), the app does not start, with a clear message.

```
0. Read configuration                                              (section 0)
1. Connect to the museum database (read-only), read the schema,    (section 1)
   check that the configured column names exist
2. Build the prompts (SQL, refine, synthesis, drafting)            (section 2)
3. Load the LLM (4-bit on a CUDA GPU, unquantized on CPU)          (section 3)
4. Load the VLM (optional; the app starts without it)              (section 4)
5. Load CLIP; load and VERIFY the FAISS index against its manifest (section 5)
6. (RRF function; annotation functions)                            (sections 6, 6b)
7. Open annotations.db (created if missing), attach it read-only   (end of section 7)
   to the museum connection, create the engine
8. Launch Gradio; show open proposals from earlier sessions        (section 8)
```

**Index verification.** The FAISS index is built by a separate Colab notebook, not by this script. At
startup the script loads it only if `index_manifest.json` proves it belongs to *this* database and *this*
CLIP model: same model name, and the same SHA-256 fingerprint over (rowid, image URL) of all eligible
records. Otherwise visual search is disabled, with the reason shown in the interface, instead of silently
returning the wrong artworks.

**Configuration values (section 0):**

| Setting | Value | What it controls |
|---|---|---|
| `RETRIEVAL_TOP_K` | 20 | Records the answer LLM reads in detail (the sample) |
| `VECTOR_SEARCH_K` | 15 | Nearest neighbours FAISS returns before thresholding |
| `VECTOR_SIMILARITY_THRESHOLD_TEXT` | 0.24 | Minimum similarity for text → image hits |
| `VECTOR_SIMILARITY_THRESHOLD_IMAGE` | 0.60 | Minimum similarity for image → image hits |
| `MAX_RESULT_ROWS_DISPLAY` | 2000 | Rows shown in "All matches"; the CSV export has all |
| `PLAUSIBLE_YEARS` | (1400, 2030) | Dates outside this range are flagged in the profile as likely errors |
| `ANNOTATIONS_PATH` | `CURRENT_DIR / "annotations.db"` | The researcher knowledge file. On Colab: must be on the mounted Drive |
| `PROPAGATION_SIMILARITY` | 0.975 | Image → image similarity for "same design" proposals |
| `PROPAGATION_SEARCH_K` | 30 | Nearest neighbours checked per annotated print |
| `PROPAGATION_MAX_PER_RULE` | 40 | A "same maker and date" rule matching more prints is skipped as too generic |
| `PROPAGATION_MAX_TOTAL` | 20 | Maximum proposals per saved insight |
| `PROPAGATION_SIMILAR_MIN` | 0.88 | Lower threshold for "visually similar" proposals (starting value) |
| `PROPAGATION_SIMILAR_MAX` | 5 | Maximum "visually similar" proposals per insight |
| `PROPAGATION_GROUNDS` | three grounds | The grounds offered in the interface |
| `PROPAGATION_DEFAULT_GROUNDS` | same design, visually similar | Ticked by default; "same maker and date" is opt-in |

Column names are also set in section 0 (`OBJECT_NUMBER_COLUMN`, `TITLE_COLUMN`, …). `CONTEXT_FIELDS`
decides which fields the answer LLM reads per record.

### 3.1 Transferring to another dataset

The prototype is built for one collection: the Rijksmuseum's early modern prints and drawings, with a Dutch
catalogue. That is deliberate. Its examples and rules use the collection's own vocabulary, because that is what
makes a small model work on it. Moving to another dataset means changing the parts below; everything else (the
engine, the annotation layer, propagation, the SQL guardrail, the checks and the logging) stays the same.

| What | Where in the script | What to change |
|---|---|---|
| Database and files | Section 0: `DATABASE_PATH`, `ANNOTATIONS_PATH` | Path of the new database; a new, empty `annotations.db` |
| Column names | Section 0: `OBJECT_NUMBER_COLUMN` … `IMAGE_COLUMN`, `CONTEXT_FIELDS` | The new table's column names, and which fields the answer model reads |
| Date sanity check | Section 0: `PLAUSIBLE_YEARS` | The range outside which dates are flagged as likely errors |
| Visual index | The Colab notebook (`build_faiss_index_colab.ipynb`) | Build a new index and manifest; at startup the script refuses an index from another database |
| Research period | Section 2: `SQL_RULES` rule 1, and the `BETWEEN` years in every few-shot example | The new collection's period (here 1450–1850) |
| Catalogue language and vocabulary | Section 2: `SQL_RULES` rule 3; section 7: `retry_sql()` ("Dutch words") | The catalogue's language, its object-type terms (here *prent*, *tekening*) and typical subject and name translations |
| How makers are written | Section 2: `SQL_RULES` rule 4; section 7: `propose_extensions()` (`"anoniem"`, `"anonymous"`) | The name format (here 'Lastname, Firstname'), the label for unattributed works, a name-variant example |
| Few-shot examples | Section 2: `SQL_FILTER_SYS` (five examples), `REFINE_SQL_SYS` (two) | Typical questions for the new collection, written with its vocabulary and columns |
| Answer prompt | Section 2: first lines of `SYNTHESIS_SYS_TEMPLATE` | The role ("expert art historian"), the collection name, the list of fields the model sees |
| Drafting example | Section 2: example in `ANNOTATION_DRAFT_SYS` | An example record in the new collection's style (title, inventory number) |
| Image description | Section 4: prompt in `analyze_image()` | Only if the collection isn't art: what the VLM should describe |
| Result profile wording | Section 7: `build_profile()` | "impressions, states or parts of one print" is print-specific |
| Propagation thresholds | Section 0: `PROPAGATION_SIMILARITY` (0.975), `PROPAGATION_SIMILAR_MIN` (0.88) | Calibrated on prints; check against the scores of a few known pairs in the new collection |
| Interface texts | Section 8: `gr.Blocks(title=…)`, the header text, placeholders | Title, description and example questions |

Two rules when doing this. First, these parts hold **collection knowledge, never case knowledge**: the death year
of one maker or the publisher of one series belongs in the annotations. Second, few-shot examples should show
**typical vocabulary and query patterns, not the questions you are going to test with**, or the test will look
better than the system is.

**Assumptions the engine makes about any dataset:** one table, with an identifier, a persistent link, a title,
an object type, one creator, a creation date (any format with a four-digit year) and an image URL. A dataset
that records several makers per object, or keeps them in a separate table, needs changes to the engine itself.

**Files on disk:**

| Path | What it is |
|---|---|
| `DATABASE_PATH` (`../preprocessing/rma_artworks`) | Museum database, opened read-only |
| `artworks.index`, `row_ids.npy`, `index_manifest.json` | FAISS index, its rowids, and the manifest that proves they match the database |
| `annotations.db` | Researcher knowledge: annotations, proposals, provenance. The only file the app writes to during a session |
| `session_log.csv` | Written by "Export session log" |
| `search_results.csv` | Written by "Export all matches" |

---

## 4. Request types (what runs per interaction)

Everything after startup happens in `ResearchEngine` (section 7), triggered by the interface (section 8).
There are four retrieval actions and three annotation actions.

### 4.1 New search → `new_search()`

```
Question and/or image
   │
   ├─ image? → analyze_image() (VLM) → caption, kept for the whole retrieval session
   │
   ├─ SQL question = typed text (+ two condensed caption keywords, only if the
   │                  experimental checkbox is on)
   │      generate_sql() → run_sql()  (safety checks; SELECT list rewritten to rowid;
   │                                   the query can include the 'annotated' view)
   │      → sql_ids
   │
   ├─ CLIP gate: image → always (threshold 0.60)
   │             text  → only if "also use visual similarity" is ticked (threshold 0.24)
   │      encode_query() → vector_search() for global hits (image mode, or when SQL found nothing)
   │
   ▼
retrieve()  → all matches, profile, sample, context (section 5)
   ▼
_finish()   → conversation history reset → synthesize() → attach_sources() → log
```

A new search always resets the retrieval state, the conversation history and the trace.

**SQL guardrail: when a query finds nothing** (`_check_empty_result()`), the code counts, for every `LIKE` condition in the
query, how many records in the whole catalogue match it on its own (`term_counts()`). Terms with 0 records
don't occur in the catalogue: typically a wrong translation ("temptatie" instead of "verzoeking") or a name
form the catalogue doesn't use ("William" instead of "Willem"). For a generated query, the model gets one
retry with these counts as feedback (`retry_sql()`); the answer then starts with a note that the system
retried, and the trace shows both queries. If every term occurs on its own, there is no retry: the note says
the combination is too narrow. The data checks the translation, instead of the model guessing why nothing
was found.

### 4.2 Refine → `refine_search()`

The LLM *edits* the previous SQL query (`REFINE_SQL_SYS`) instead of writing a new one, so earlier
conditions (e.g. the creator) are kept. The vector half of the last search is reused. If the refined query
returns records that were *not* in the previous result set, the answer starts with a warning: the
refinement widened or changed the search instead of narrowing it. If it finds nothing, the term counts are
shown (no retry).

### 4.3 Follow-up → `follow_up()`

No retrieval. The answer LLM reasons over the same sample, with the conversation history. If nothing has
been retrieved yet, it falls back to a new search. After an annotation is saved or a proposal is
confirmed, the context is rebuilt, so a follow-up already sees it.

### 4.4 Re-run edited SQL → `rerun_edited_sql()`

The researcher edits the SQL box and re-runs it. Same safety checks; the vector half of the last search is
reused; the conversation resets. A rejected query leaves the engine state untouched. If it finds nothing,
the term counts are shown; the researcher's own query is never rewritten.

### 4.5 Draft with AI → `draft_annotation()`

The researcher describes an insight in their own words. The LLM (`ANNOTATION_DRAFT_SYS`, deterministic,
200 tokens) returns JSON: artworks, label, note. The code reads it defensively (first `{...}` block; numbers
like `"#2"` accepted) and keeps only artwork numbers that are actually in the context. The draft fills the
panel; **nothing is saved**. Both the researcher's words and the raw draft are logged (`draft_annotation`).

### 4.6 Save insight → `save_annotation()` → `propose_extensions()`

```
Selected artworks + label + note + chosen grounds
   │
   ├─ add_annotation()          origin 'researcher', status 'confirmed'
   ├─ _refresh_annotations()    tables get the 'annotations' column
   ├─ _build_context()          the LLM's context includes it immediately
   ├─ log: annotate
   │
   └─ propose_extensions()      only on the grounds the researcher chose (section 6.3)
          → proposals saved as origin 'system', status 'proposed'
          → log: propose (with the grounds)
```

### 4.7 Confirm / reject proposals → `resolve_proposals()`

Sets the status of the selected proposals. Confirmed proposals become knowledge (tables, context, SQL view);
rejected ones are kept with status `rejected` and are never proposed again for the same insight. Logged as
`confirm_proposals` / `reject_proposals`. Unticked proposals stay open, also across sessions.

---

## 5. Results: three levels and the context

`retrieve()` builds three levels, and keeping them apart is the core of the design:

| Level | What it is | Produced by |
|---|---|---|
| **All matches** (`self.results`) | Every record of the result set, sorted by year (`_year()` takes the first four-digit year, so `1525-04-11` and `ca. 1570` sort correctly) | SQL (as a subquery, so size doesn't matter) plus vector hits |
| **Profile** (`self.profile`) | Exact facts about all matches: count, distinct titles, date range and per-period counts, implausible dates, object types, top creators, image and index coverage | `build_profile()`, code only |
| **Sample** (`self.context_df`) | At most 20 records the LLM reads, numbered "Artwork #1 …" | A fixed rule, stated in the ranking note |

**How the sample is chosen** (the rule is shown to the researcher and given to the LLM):

| Situation | Sample |
|---|---|
| Text question, SQL found rows, no query vector | Evenly spread over the chronology of all matches (`chronological_sample()`) |
| Text question, SQL found rows, CLIP ticked | The SQL matches most similar to the question (CLIP only ranks; it adds nothing) |
| Text question, SQL found nothing | CLIP's global hits |
| Image uploaded | SQL matches (ranked by similarity to the image) and CLIP's global hits, fused with RRF |

**The context the LLM reads** (`_build_context()`), one line per record:

```
[Artwork #2 | RP-P-1888-A-12477] title: Presentatie van Maria | object type: prent | creator: Cort, Cornelis
| creation date: 1602 | annotated: A5 (Orlandi reissue)
```

followed by one block with each annotation once:

```
RESEARCHER ANNOTATIONS (the researcher's own confirmed knowledge, not catalogue data):
A5 on Artwork #2 (system, confirmed; proposed by the system because: same maker and date: Cort, Cornelis,
1602): Orlandi reissue: Published by Orlandi
```

Image URLs and empty values are left out. The context contains catalogue fields and confirmed annotations
only; everything else the model is told about the data (the profile, how the sample was chosen, how many
annotations are present) is added to the system prompt (section 8).

---

## 6. The annotation layer (section 6b and engine)

### 6.1 Storage

`annotations.db` is a separate, writable SQLite file. The museum database is never written to.

```
annotations         id, label, note, origin ('researcher'|'system'),
                    status ('proposed'|'confirmed'|'rejected'),
                    based_on (the insight a proposal extends), reason (why the system proposed it),
                    created_at, updated_at
annotation_records  annotation_id, rowid_ref        -- one row per print it applies to
```

- One annotation can cover several prints (a many-to-many relation).
- `rowid_ref`, not `rowid`: every SQLite table already has a hidden `rowid` column.
- The `reason` column is added with `ALTER TABLE` when missing, so older annotation files are upgraded.
- **Only confirmed annotations count as knowledge** (`all_annotations()`). Proposals are invisible to the
  LLM and to SQL until confirmed.

### 6.2 How annotations reach search and answers

| Where | How |
|---|---|
| **SQL** | `attach_annotations()` attaches the file *read-only* to the museum connection, with a temporary view `annotated(rowid_ref, label, note, origin)` of confirmed annotations. SQL rule 6 and a few-shot example teach the model to use `rowid IN (SELECT rowid_ref FROM annotated WHERE label LIKE '%term%' OR note LIKE '%term%')`. The attach happens in code; `attach` stays on the blocked keyword list for generated SQL |
| **Tables** | An `annotations` column ("A1: Orlandi reissue") in both result tables |
| **Context** | Inline label per record, plus the annotations block (section 5). The synthesis prompt names how many annotations are present |
| **Below the answer** | Cited annotations in the sources, and the code-written "Known about these records" block (section 7) |

### 6.3 Propagation (`propose_extensions()`)

After an insight is saved, the system proposes other prints it may apply to, on the grounds the researcher
ticked for this insight (same design and visually similar by default):

| Ground | How | Finds | Cannot |
|---|---|---|---|
| **Same design** (visually near-identical) | FAISS neighbours of each annotated print, similarity ≥ 0.975 | Other impressions, states, copies | Separate impressions by date: a 1570 and a 1602 impression from one plate look identical |
| **Visually similar** (related composition or motif) | The 5 closest FAISS neighbours with similarity ≥ 0.88, not already proposed as same design | Related compositions, borrowed motifs, possible influence | Tell influence from a shared subject: CLIP also scores two unrelated scenes of the same subject as similar |
| **Same maker and date** (opt-in) | Exact match on creator and creation date in the metadata | E.g. a reissue campaign (all Cort prints dated 1602) | Find near dates (1605), or work for 'anonymous' (skipped), or very common combinations (> 40, skipped); outside reissues mostly noise |

Prints that already carry the insight, or were proposed for it before (also rejected ones), are skipped.
Each proposal stores its ground in `reason`, which is shown to the researcher and, once confirmed, to the
LLM. Not every insight travels the same way: a copy follows the design, an influence looks similar, a reissue
may follow the date, an attribution may follow none of them. Hence the per-insight choice.

Why these defaults: in the first two test sessions, "same maker and date" was right for one reissue campaign
(15 of 15 confirmed) and mostly noise for other insights (3 of 30 confirmed), while "same design" found
nothing for insights about similarity or influence, which is what the researcher annotated in new cases.

### 6.4 Lifecycle of a piece of knowledge

```
researcher writes (or drafts with AI, then edits)  →  annotate  (researcher, confirmed)
        │
        └→ system proposes on chosen grounds        →  propose  (system, proposed, reason, based_on)
                │
                ├→ researcher confirms               →  confirm_proposals (system, confirmed)
                ├→ researcher rejects                →  reject_proposals  (system, rejected; never re-proposed)
                └→ left open                         →  stays 'proposed', shown again next session
```

---

## 7. Traceability: what you can inspect, and where

**Below every answer** (`attach_sources()`, code-written):

1. **Scope line**: how many records the answer is based on, and how they were chosen.
2. **Sources**: each cited artwork with inventory number, title, creator, date and persistent link; each
   cited annotation with origin, status and the artworks it is on.
3. **Warnings**: a cited artwork or annotation that isn't in the context (also bare "A6" mentions), or an
   answer that cites no records at all.
4. **Known about these records**: every annotation on the records in the context, grouped by label, with
   artworks, IDs, origin and the researcher's note. Groups the answer didn't mention are marked
   **[not mentioned in the answer]**, so the gap between what is known and what was said is visible.

**Elsewhere:**

| Where | What it shows |
|---|---|
| Result tables | `Artwork #`, `found_via` (SQL / Vector / both), `vector_similarity`, `annotations` |
| Gallery | Each image captioned "Artwork #N (ID …) - title" |
| Profile box | Exact facts about all matches |
| VLM box | The literal caption of an uploaded image |
| Retrieval trace | Every step of the last action (see below) |
| Session log / CSV | Every action of the session (see below) |

**Trace lines worth knowing:** `[SQL]` generated query and match count; `[Vector]` candidates and scores;
`[Terms]` term counts and the retry when a query found nothing; `[Rank]` how the sample was chosen; `[Annotations]` how many records in the context carry annotations,
saves, drafts; `[Propagation]` proposals per insight, skipped rules, confirmations; `[Check]` citation
warnings and "Annotations in the context: N; mentioned in the answer: M".

**Session log columns:** timestamp, mode, query, sql, answer, total_matches, ranking_note, context_records,
context_annotations (what the model had), cited_annotations (what it mentioned), llm.

**Modes in the log:** `new_search`, `refine`, `follow_up`, `rerun_sql`, `draft_annotation`, `annotate`,
`propose`, `confirm_proposals`, `reject_proposals`. Together they give the full provenance chain of every
piece of knowledge. Comparing `context_annotations` with `cited_annotations` measures how well the model
uses annotations.

---

## 8. The prompts

| Prompt | Used by | Purpose |
|---|---|---|
| `SQL_RULES` | both SQL prompts | Six rules: scope 1450–1850; exact bracketed column names; Dutch terms and Dutch name forms, AND between different elements of the question, OR only for alternatives of one element; 'Lastname, Firstname' matching, name variants, parentheses per maker; image descriptions as hints; annotations only via the `annotated` view, never invented columns |
| `SQL_FILTER_SYS` | `generate_sql()` | Question → one query, with five few-shot examples (one with AND between elements, one using `annotated`) |
| `REFINE_SQL_SYS` | `refine_sql()` | Edit the previous query as little as possible |
| `SYNTHESIS_SYS_TEMPLATE` | `synthesize()` | Answer / Observations / Limits for a new question; direct answer for a follow-up; every claim cites "Artwork #N"; observations must compare records or use an annotation; no claims about what the images look like; say so when the records don't match the question, never reinterpret a title to fit; never claim something is absent from the database; annotations used already in the Answer and cited as "Annotation A&lt;n&gt;" |
| `ANNOTATION_DRAFT_SYS` | `draft_annotation()` | Restructure the researcher's words only: never add facts, never make the note more specific, no "Artwork #" numbers in the note, all artworks only if the researcher says so; neutral examples |

At run time the synthesis prompt also gets: the profile (for any statement about the whole result set), a
note when only part of the matches is in the context, the number of annotations present, the VLM caption
(labelled as a rough aid), and whether this is a new or a follow-up question.

**Generation settings** (all through `generate_chat()`):

| Call | temperature | repetition_penalty | max_new_tokens | Other |
|---|---|---|---|---|
| `generate_sql()` | 0.0 | 1.0 | 250 | stops at the first `;` |
| `refine_sql()` | 0.0 | 1.0 | 300 | stops at the first `;` |
| `retry_sql()` | 0.0 | 1.0 | 250 | stops at the first `;`; gets the term counts |
| `extract_filter_hints()` | 0.0 | 1.0 | 15 | experimental |
| `draft_annotation()` | 0.0 | 1.0 | 200 | JSON |
| `synthesize()` | 0.3 | 1.05 | 900 | top_p 0.9, top_k 50 |

`repetition_penalty` also penalises tokens already in the prompt, so tasks that copy from it (SQL, JSON)
keep it at 1.0; looping is prevented by the stop string instead.

---

## 9. Engine state reference

**Retrieval state** (overwritten by new search, refine and re-run; kept by follow-up):
`results`, `context_df`, `context_str`, `images`, `profile`, `ranking_note`, `image_analysis`, and
`last_sql`, `last_sql_ids`, `last_question`, `last_vector_hits`, `last_query_vec`, `last_search_was_image`
(so refine and re-run can reuse the vector half without re-running CLIP).

**Annotation state** (refreshed after every retrieval, save and confirmation):
`annotations_by_rowid` (all confirmed annotations per rowid), `context_annotations` (those in the current
context, with the artworks they're on), `last_cited_annotations` (what the last answer mentioned).

**Fixed lookup:** `rowid_to_pos` (rowid → position in the FAISS index).

**Conversation state:** `history`, reset by every retrieval, extended by every answer.

**Trace state:** `_trace_buffer`, reset at the start of every search, refine, follow-up and re-run.

**Persistent:** `session_log` (for the life of the process) and `annotations.db` (across sessions).

---

## 10. Known limitations

**The answer model**
- **Model size is the main lever.** With the same data, prompt and annotation, Qwen2.5-1.5B confused a
  1602 reissue with the 1570 original and ignored the annotation; Qwen2.5-7B answered correctly and cited
  it. The 1.5B is fine for testing the interface, not for judging whether co-exploration works.
- **It doesn't volunteer annotations.** In testing, the 7B used annotations reliably when the question
  pointed at them ("Orlandi", "Goltzius"), but not for open questions ("what do we know about these
  prints?"), where it cited none of the 12 in its context. The "Known about these records" block makes
  them visible regardless; `cited_annotations` in the log measures it.
- **It reads dates literally.** The data has no birth or death dates of makers, so a print dated 1602 under
  a maker who died in 1578 can be described as the maker at work in 1602. An annotation on such prints
  ("later impression", "reissue") corrects this for the records it covers.
- **It still invents occasionally**: annotation IDs numbered after artworks (warned), style claims despite
  the rule, statements contradicting the profile, padded observations. In the transfer test it also made
  unrelated titles fit the question and claimed a maker was absent from the database; the prompt now rules
  both out, which still needs retesting. The code-written sources, warnings
  and profile are there to catch this; the answer is the least reliable part of the output.
- **No KV-cache reuse**: each call reprocesses the full prompt; long follow-up conversations get slower.

**Retrieval**
- **Only the sample reaches the answer** (at most 20 records). Statements about the whole result set
  should come from the profile.
- **Retrieval is the bottleneck in new cases, and translation is its weakest step.** In the first transfer
  test the SQL model combined different elements of a question with OR (1,067 matches). With AND enforced, the
  second test showed the underlying problem: the 7B model often doesn't find the Dutch words the catalogue uses
  ("begraafprocesjon", "temptatie", "William", "Sint" for *heilige*/*H.*), so a query finds nothing instead of
  too much. The term check and one retry make this visible and often recoverable; editing the SQL remains the
  reliable fallback. A larger model, or semantic search over the titles, would address it at the root.
- **The sample shifts when the result set changes.** The 20 records the model reads are spread evenly over
  the chronology of all matches, so one record moving (e.g. after the date fix) changes which records are
  picked. A work can be among the matches ("All matches" table) without being in the sample or the gallery.
- **Annotations are found through SQL text matching** (`LIKE` on label and note). This depends on the model
  writing the subquery, and is literal: "not published by Lafreri" could become `note NOT LIKE '%Lafreri%'`
  and exclude Orlandi reissues whose note says "from Lafreri's stock".
- **Visual search covers only the indexed prints and drawings**; check the coverage line in the profile.
- **In image mode, typed text does not restrict the results**: SQL and CLIP are fused as two ranked lists.
- **CLIP nearest neighbours are not attribution**, and the similarity thresholds are starting points, not
  calibrated cutoffs.

**Annotations and propagation**
- **Propagation rules are simple.** "Same design" can't tell impressions apart by date; "visually similar"
  can't tell influence from a shared subject; "same maker and date" needs an exact date match (the 1605
  impressions in the Orlandi test were missed) and skips anonymous and very common combinations.
- **`PROPAGATION_SIMILAR_MIN` (0.88) is a starting value**, not yet checked on real proposals; the scores are
  in each proposal's reason, so it can be tuned from use.
- **`PROPAGATION_SIMILARITY` (0.975) was calibrated on one case** (Cort's landscape series, where 0.97–0.98
  separated designs). Check it on other material.
- **Annotations attach to prints, not to makers.** Knowledge about a maker (dates, workshop, active years,
  name variants) has to be recorded print by print. Maker-level annotations would let the system flag, for any maker,
  prints dated after the maker's death or attributions that cannot be right.
- **Annotations are only as good as their input.** The system doesn't check them against the data; e.g. it
  has no birth or death dates of makers, so it cannot flag a chronologically impossible attribution.
- **Reject means "does not apply"** and is permanent for that insight; unsure proposals should stay unticked.
- **Artwork numbers are not stable references.** "Artwork #5" means a different print in another search. In the
  second test an insight was saved on the wrong prints this way (the same sentence reused from an earlier
  session). Annotations themselves always store the prints (rowids), but the interface has no way to correct or
  retract one: a wrong annotation has to be fixed in `annotations.db` directly. Check the ticked artworks before
  saving.

**Data**
- One creator per record; designer, engraver and publisher are not distinguished.
- Records with the same title can be impressions, states or parts of one print.
- Dates after a maker's death usually refer to later impressions or editions.
- A few records carry implausible dates; the profile lists them.

**Technical**
- `main_table = tables[0]` assumes a single-table museum database.
- The dangerous-keyword check in `run_sql()` is a word filter, not a parser; the real safeguard is the
  read-only connection (and the read-only attach of the annotations).
- SQLite on a mounted Google Drive works for one user, but is slow; copy the museum database to `/content`
  if needed, but keep `annotations.db` on Drive.
- 4-bit loading needs CUDA.

---

## 11. Changes since the previous version of this document

- **Retrieval and results**: `rank_and_fuse` replaced by `retrieve()` with three levels (all matches,
  profile, sample); sample of 20 instead of 7, chosen by an explicit rule; refine mode; scope line, sources
  and citation warnings written by code; FAISS index built by a notebook and verified by a manifest.
- **Annotation layer** (new): `annotations.db` with provenance; save insight; draft with AI; propagation on
  two grounds, chosen per insight; confirm/reject; proposals persist across sessions.
- **Annotations in search and answers** (new): read-only `annotated` view for SQL; annotations column in
  the tables; inline labels and an annotations block in the context.
- **Checks and logging** (new): cited annotations in the sources; warnings for invented annotation IDs;
  "Known about these records" block; `context_annotations` and `cited_annotations` in the session log;
  full path of `annotations.db` printed at startup.
- **Prompts**: no claims about images; observations must compare records or use annotations; never invent
  columns; neutral drafting example.
- **Clean-up**: case-specific knowledge removed from the code (`MAKER_DEATH_YEARS` and the computed
  "posthumous" note it produced, Orlandi examples in the drafting prompt and the interface); the synthesis
  prompt states the no-images rule once; the annotation-citation check runs once per answer; an answer-less
  search no longer logs the previous answer's cited annotations; `analyze_image()` re-indented.
- **After the transfer test** (Goltzius, Hogenberg, Verheyden/Bosch): full dates such as `1525-04-11` are read
  correctly (`_year()`), instead of being sorted last; a third propagation ground, "visually similar"; "same
  maker and date" no longer ticked by default; SQL rules for AND/OR grouping, Dutch name forms, name variants
  and multiple makers, with a new few-shot example; answers must say when records don't match the question and
  may not claim absence from the database. Examples use neutral terms, not those of the test sessions, so a
  retest isn't flattered.
- **After the second transfer test**: the SQL guardrail (term counts when a query finds nothing, with one
  retry for generated queries and counts only for the researcher's own SQL). A feature to correct and retract
  annotations was built and removed again, because it added a management layer for an edge case.
- **Version 1.0**: clean-up before freezing. The code stays domain-specific; what would change for another
  dataset is documented in section 3.1 instead of being moved into the code.
