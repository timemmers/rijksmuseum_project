"""
Rijksmuseum graphics arts AI assistant - A text-to-SQL and multimodal vector search AI system with Gradio UI
---
python -m pip install torch transformers faiss-cpu accelerate numpy gradio pillow sentence-transformers pandas bitsandbytes
"""

from pathlib import Path
import re
import sqlite3
from typing import Optional
import csv
import hashlib
import json

from PIL import Image
import numpy as np
import faiss
import datetime
import pandas as pd

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoProcessor, \
    AutoModelForImageTextToText, BitsAndBytesConfig
import gradio as gr
import gc


# 0. CONFIGURATION (change LLMs according to computing power)

LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# e.g. Qwen3.5
VLM_MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"
# e.g. Qwen2.5-VL
CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
# must be the model the FAISS index was built with

ENABLE_VLM = True # set to False on low-resource machines: image searches then run on CLIP only

CURRENT_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
DATABASE_PATH = CURRENT_DIR.parent / "preprocessing" / "rma_artworks"

RETRIEVAL_TOP_K = 20 # max number of records the answer LLM reads in detail (scale up with bigger models)
VECTOR_SEARCH_K = 15
MAX_RESULT_ROWS_DISPLAY = 2000 # rows shown in the 'All matches' table; the CSV export always contains all of them

# Text -> image and image -> image CLIP similarities live on very different scales, so each mode gets its own threshold
VECTOR_SIMILARITY_THRESHOLD_TEXT = 0.24 # Minimum cosine similarity to drop weak matches before RRF pollution
VECTOR_SIMILARITY_THRESHOLD_IMAGE = 0.60 # Image -> image scores run much higher; tune using the scores in the trace

# Column names in the database (the script stops at startup and lists the real names if one is wrong)
OBJECT_NUMBER_COLUMN = "objectInventoryNumber"
HANDLE_COLUMN = "objectPersistentIdentifier"
TITLE_COLUMN = "objectTitle[1]"
TYPE_COLUMN = "objectType[1]"
CREATOR_COLUMN = "objectCreator[1]"
DATE_COLUMN = "objectCreationDate[1]"
IMAGE_COLUMN = "objectImage"

# Fields the answer LLM reads per artwork, with short labels (fewer tokens, easier for small models)
CONTEXT_FIELDS = [(TITLE_COLUMN, "title"), (TYPE_COLUMN, "object type"), (CREATOR_COLUMN, "creator"),
                  (DATE_COLUMN, "creation date")]
PLAUSIBLE_YEARS = (1400, 2030) # creation dates outside this range are flagged in the profile as likely data errors


# 1. DATABASE SETUP & SCHEMA EXTRACTION

def connect_and_get_schema(db_path: Path):
    """
    Connects to SQLite database (read-only), detects existing tables,
    and constructs a readable schema string for Text-to-SQL prompting
    """

    # sqlite3.connect silently creates an empty database for a missing path, so check first
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # mode=ro: no generated query can modify the database, whatever the LLM writes
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]

    main_table = tables[0]
    db_schema = {}

    for table_name in tables:
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        db_schema[table_name] = [(col_info[1], col_info[2]) for col_info in cursor.fetchall()]

    # Stop early with a clear message if a configured column name doesn't exist
    columns = [name for name, _ in db_schema[main_table]]
    missing = [c for c in (OBJECT_NUMBER_COLUMN, HANDLE_COLUMN, TITLE_COLUMN, TYPE_COLUMN, CREATOR_COLUMN,
                           DATE_COLUMN, IMAGE_COLUMN) if c not in columns]
    if missing:
        raise ValueError(f"Column(s) {missing} not found in '{main_table}'. Available columns: {columns}. "
                         f"Set the right names in section 0.")

    schema_str = "".join(f"Table: {t}\nColumns: {', '.join(f'{n} ({typ})' for n, typ in cols)}\n\n"
                         for t, cols in db_schema.items())

    print(f"[Database] Connected to '{db_path}' (read-only). Primary table detected: '{main_table}'")
    return conn, main_table, schema_str

conn, MAIN_TABLE, SCHEMA_STR = connect_and_get_schema(DATABASE_PATH)


# 2. SYSTEM PROMPTS

SQL_RULES = """CRITICAL RULES:
1. Scope: the database covers all periods, but this tool is built for early modern prints and drawings. Always filter
   "objectCreationDate[1]" BETWEEN '1450' AND '1850', or a narrower period if the question asks for one
   (e.g. 'after 1575' -> BETWEEN '1576' AND '1850'). Only use other years if the question explicitly asks for them.
2. Use EXACT column names with brackets, e.g. "objectType[1]", "objectCreator[1]", "objectCreationDate[1]".
3. The database uses Dutch terms. Translate English search terms into Dutch for LIKE clauses.
   Examples: 'print' -> 'prent', 'drawing' -> 'tekening', 'landscape' -> 'landschap', 'portrait' -> 'portret'.
   Subjects can only be found through the Dutch title, and titles vary, so combine several Dutch terms with OR
   inside parentheses, e.g. landscape -> ("objectTitle[1]" LIKE '%landschap%' OR "objectTitle[1]" LIKE '%zicht%').
4. Creator names are stored as 'Lastname, Firstname' (e.g. 'Cort, Cornelis'). Never match a full name as one string.
   Instead: WHERE "objectCreator[1]" LIKE '%Cornelis%' AND "objectCreator[1]" LIKE '%Cort%'
   Unattributed works have the creator 'anonymous'.
5. If the question includes a visual description (from an uploaded image) rather than explicit search terms,
   infer plausible object type / subject / period filters from it the same way you would from a text question.
"""

SQL_FILTER_SYS = f"""You are an expert AI assistant that translates natural language questions into executable SQLite SQL queries.

Database schema:
{SCHEMA_STR}

{SQL_RULES}
FEW-SHOT EXAMPLES:

Question: Which prints were produced by Cornelis Cort?
SQL: SELECT rowid FROM {MAIN_TABLE} WHERE "objectType[1]" LIKE '%prent%' AND "objectCreator[1]" LIKE '%Cornelis%' AND "objectCreator[1]" LIKE '%Cort%' AND "objectCreationDate[1]" BETWEEN '1450' AND '1850';

Question: Show me drawings of landscapes from the 16th century
SQL: SELECT rowid FROM {MAIN_TABLE} WHERE "objectType[1]" LIKE '%tekening%' AND ("objectTitle[1]" LIKE '%landschap%' OR "objectTitle[1]" LIKE '%zicht%') AND "objectCreationDate[1]" BETWEEN '1501' AND '1600';

Question: Show me 17th century portrait prints
SQL: SELECT rowid FROM {MAIN_TABLE} WHERE "objectType[1]" LIKE '%prent%' AND "objectTitle[1]" LIKE '%portret%' AND "objectCreationDate[1]" BETWEEN '1601' AND '1700';

Return ONLY the raw SQL query ending with a semicolon, nothing else."""

# Refining edits the previous query instead of writing a new one, so earlier conditions (e.g. the creator) are kept
REFINE_SQL_SYS = f"""You are an expert AI assistant that edits executable SQLite SQL queries.
Change the previous query as little as possible: keep ALL its conditions and only add or adjust what the
refinement asks for.

Database schema:
{SCHEMA_STR}

{SQL_RULES}
FEW-SHOT EXAMPLES:

Previous SQL: SELECT rowid FROM {MAIN_TABLE} WHERE "objectType[1]" LIKE '%prent%' AND "objectCreator[1]" LIKE '%Cornelis%' AND "objectCreator[1]" LIKE '%Cort%' AND "objectCreationDate[1]" BETWEEN '1450' AND '1850';
Refinement: Only give those after 1575
SQL: SELECT rowid FROM {MAIN_TABLE} WHERE "objectType[1]" LIKE '%prent%' AND "objectCreator[1]" LIKE '%Cornelis%' AND "objectCreator[1]" LIKE '%Cort%' AND "objectCreationDate[1]" BETWEEN '1576' AND '1850';

Previous SQL: SELECT rowid FROM {MAIN_TABLE} WHERE "objectType[1]" LIKE '%prent%' AND "objectCreationDate[1]" BETWEEN '1601' AND '1700';
Refinement: only the portraits
SQL: SELECT rowid FROM {MAIN_TABLE} WHERE "objectType[1]" LIKE '%prent%' AND "objectCreationDate[1]" BETWEEN '1601' AND '1700' AND "objectTitle[1]" LIKE '%portret%';

Return ONLY the complete refined SQL query ending with a semicolon, nothing else."""

SYNTHESIS_SYS_TEMPLATE = """You are an expert art historian assisting a researcher in exploring the Rijksmuseum's
early modern print and drawing collection.

You are given a set of retrieved artwork records as context. Use ONLY this context and the
conversation so far — do not invent facts that are not present in it. You see catalogue metadata
(title, object type, creator, creation date), not the images themselves.

For a NEW question, answer in this structure:
1. Answer: 1-3 sentences that answer the question directly. Take every number, date range or
   distribution about the whole result set from the exact facts about all matches, never from
   counting the records in the context.
2. Observations: 2-4 points about the records in the context (themes, series, chronology, contrasts,
   outliers), each starting with the artwork it is based on, e.g. "Artwork #4 (1559): ...".
3. Limits: 1-2 sentences on what these records cannot show (e.g. metadata only, or only a sample).
For a FOLLOW-UP question, answer it directly without this structure, but still cite every claim as "Artwork #N".

Guidelines:
- If the context doesn't contain enough information to answer confidently, say so explicitly
  rather than guessing.
- If a question concerns records that are not in the context (e.g. a subset of all matches),
  say that a refined or new search is needed instead of answering from the context.
- A date in a title usually refers to the depicted event or place, not to when the object was made
  (e.g. a battle of 1693 in a print made in 1702); use the creation date for the object itself.
- Write for a researcher: precise, willing to flag uncertainty, comfortable making a reasoned
  comparative judgment rather than only describing.

Database Context (retrieved artworks for this session):
{context}
"""


# 3. LLM SETUP

print(f"[LLM] Loading {LLM_MODEL_NAME}...")
# 4-bit quantization (bitsandbytes) needs a CUDA GPU; on CPU the model loads unquantized
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
) if torch.cuda.is_available() else None
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"[LLM] {LLM_MODEL_NAME} loaded!")


def generate_chat(messages:list, max_new_tokens: int = 500, temperature: float = 0.0,
                  top_p: float = 1.0, top_k: int = 50, repetition_penalty: float = 1.0,
                  stop_strings: Optional[list] = None) -> str:
    # Note: repetition_penalty also penalizes every token already in the PROMPT (schema, context),
    # so tasks that must copy from the prompt (SQL) keep it at 1.0
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    device = next(llm_model.parameters()).device
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    do_sample = temperature > 0.0
    gen_cfg = GenerationConfig(
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        stop_strings=stop_strings,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_cfg.temperature = temperature
        gen_cfg.top_p = top_p
        gen_cfg.top_k = top_k
    else:
        # Greedy decoding: clear the sampling settings, otherwise the model's own defaults are merged in
        # and transformers warns that they are ignored
        gen_cfg.temperature = gen_cfg.top_p = gen_cfg.top_k = None

    # The tokenizer is passed so generate() can detect stop_strings
    output_ids = llm_model.generate(input_ids, attention_mask=attention_mask, generation_config=gen_cfg,
                                    tokenizer=tokenizer)
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


# 4. VLM SETUP

vlm_processor = None
vlm_model = None
if ENABLE_VLM:
    try:
        print(f"[VLM] Loading Vision Language Model ({VLM_MODEL_NAME})...")
        vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME)
        vlm_model = AutoModelForImageTextToText.from_pretrained(
            VLM_MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print(f"[VLM] {VLM_MODEL_NAME} loaded successfully!")
    except Exception as vlm_err:
        print(f"[VLM Warning] Could not load VLM: {vlm_err}")
else:
    print("[VLM] Disabled in the configuration (ENABLE_VLM = False).")


def analyze_image(user_image: Image.Image) -> str:
        """
        Analyzes an uploaded image using the VLM
        """

        if user_image is None or vlm_model is None or vlm_processor is None:
            return ""

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                "Describe this artwork for an art-historical comparison. "
                                "Cover: likely medium/technique (e.g. engraving, etching, chalk drawing), "
                                "line quality, composition type (portrait, landscape, scene), subject matter, "
                                "and any stylistic cues to period or origin. Be specific, not generic."
                            ),
                        },
                    ],
                }
            ]
            prompt = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = vlm_processor(text=prompt, images=[user_image], return_tensors="pt")
            device = next(vlm_model.parameters()).device
            inputs = inputs.to(device)

            # Tweak these parameters for more (or less) creative descriptions; e.g do_sample false
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                generated_ids = vlm_model.generate(
                    **inputs,
                    max_new_tokens=220,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.85,
                    repetition_penalty=1.1,
                )

            new_tokens = generated_ids[:, input_len:]
            text = vlm_processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            print(f"[VLM] Description generated: {text}")
            return text
        except Exception as e:
            print(f"[VLM] Error: {e}")
            return ""



# 5. CLIP EMBEDDINGS + FAISS INDEX
# The index is built with the Colab notebook (build_faiss_index_colab.ipynb), not by this script

print(f"[Embeddings] Loading CLIP model ({CLIP_MODEL_NAME})...")
embedding_model = SentenceTransformer(CLIP_MODEL_NAME)

INDEX_FILE = CURRENT_DIR / "artworks.index"
ROW_IDS_FILE = CURRENT_DIR / "row_ids.npy"
MANIFEST_FILE = CURRENT_DIR / "index_manifest.json"

def records_fingerprint(db_conn, table_name: str, where_sql: str, image_col: str) -> str:
    """
    SHA-256 over (rowid, image URL) of all records the index should contain. The build notebook runs the
    same code, so equal fingerprints mean the index's row ids point to the same records and images
    """

    h = hashlib.sha256()
    for rid, url in db_conn.execute(f'SELECT rowid, "{image_col}" FROM {table_name} WHERE {where_sql} ORDER BY rowid'):
        h.update(f"{rid}|{url}\n".encode("utf-8"))
    return h.hexdigest()

def load_faiss_index(db_conn, embed_model):
    """
    Loads the FAISS index, but only if its manifest proves it belongs to THIS database and CLIP model.
    Otherwise visual search is disabled (with the reason) instead of silently returning wrong artworks
    Returns (index, row_ids, status message)
    """

    dim = len(embed_model.encode("dimension check")) # 512 for ViT-B/32, 768 for ViT-L/14
    empty = (faiss.IndexFlatIP(dim), np.array([], dtype=int))

    if not (INDEX_FILE.exists() and ROW_IDS_FILE.exists() and MANIFEST_FILE.exists()):
        return (*empty, f"Visual search disabled: no complete index in '{CURRENT_DIR.resolve()}' (expected "
                        f"artworks.index, row_ids.npy and index_manifest.json from the Colab notebook).")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest["clip_model"] != CLIP_MODEL_NAME:
        return (*empty, f"Visual search disabled: the index was built with {manifest['clip_model']}, "
                        f"but CLIP_MODEL_NAME is {CLIP_MODEL_NAME}.")

    print("[FAISS] Checking that the index matches this database...")
    fingerprint = records_fingerprint(db_conn, manifest["table"], manifest["eligibility_where"],
                                      manifest["image_column"])
    if fingerprint != manifest["records_fingerprint"]:
        return (*empty, "Visual search disabled: the index was built from a different version of the database. "
                        "Rebuild it with the Colab notebook.")

    index = faiss.read_index(str(INDEX_FILE))
    row_ids = np.load(str(ROW_IDS_FILE))
    if index.ntotal != len(row_ids) or index.ntotal != manifest["n_vectors"] or index.d != dim:
        return (*empty, "Visual search disabled: the index files are incomplete or don't belong together.")

    status = (f"Visual index verified: {index.ntotal:,} of {manifest['n_eligible']:,} eligible prints and drawings "
              f"({manifest['n_failed']:,} images could not be downloaded), built {manifest['built_at']}.")
    return index, row_ids, status


faiss_index, faiss_row_ids, INDEX_STATUS = load_faiss_index(conn, embedding_model)
print(f"[FAISS] {INDEX_STATUS}")


# 6. RECIPROCAL RANK FUSION (RRF)

def reciprocal_rank_fusion(sql_ids: list, vector_ids: list, k: int = 60, top_n: int = RETRIEVAL_TOP_K) -> list:
    """
    Combines ranked lists from SQL and vector search using Reciprocal Rank Fusion
    Score(d) = sum(1 / (k + rank_i))
    """

    scores = {}

    for rank, doc_id in enumerate(sql_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    return sorted(scores.keys(), key=lambda d:scores[d], reverse=True)[:top_n]


# 7. PROTOTYPE ENGINE

DANGEROUS_KEYWORDS = ["drop", "delete", "update", "insert", "alter", "create", "truncate", "attach", "pragma"]
# Word boundaries (\b) so that e.g. a creator named 'Walter' doesn't trigger 'alter'
DANGEROUS_PATTERN = re.compile(r"\b(" + "|".join(DANGEROUS_KEYWORDS) + r")\b", re.IGNORECASE)


def _preview_ids(ids: list, n: int = 50) -> str:
    """
    Shortens long rowid lists for the trace
    """
    return str(ids) if len(ids) <= n else f"{ids[:n]} ... (+{len(ids) - n} more)"


def _is_filled(value) -> bool:
    return pd.notna(value) and str(value).strip() not in ("", "<null>")


class ResearchEngine:
    """
    Holds retrieval state (current results) and conversation history separately,
    so a follow-up question can reason over the same retrieved artworks without
    re-running SQL/vector search, and a "new search" explicitly resets both

    Results are kept at two levels:
    - self.results: ALL records that matched (the 'All matches' table, the profile and the CSV export)
    - self.context_df: the sample of those records the answer LLM reads in detail ("Artwork #1" ...)
    """

    def __init__(self, db_conn, table_name, faiss_idx, row_ids_arr, embed_model):
        self.db_conn = db_conn
        self.table_name = table_name
        self.faiss_index = faiss_idx
        self.row_ids = row_ids_arr
        self.embed_model = embed_model

        # rowid -> position in the FAISS index, so any set of rows (e.g. SQL hits) can be scored directly
        self.rowid_to_pos = {int(r): i for i, r in enumerate(row_ids_arr)}

        # Retrieval state (persists across follow-up turns)
        self.results = pd.DataFrame()
        self.context_df = pd.DataFrame()
        self.context_str = ""
        self.images = []
        self.profile = ""
        self.ranking_note = "" # how the sample was chosen; told to the answer LLM

        # Kept so refinements and edited SQL can build on the last search without re-running CLIP
        self.last_sql = ""
        self.last_sql_ids = []
        self.last_question = ""
        self.last_vector_hits = []
        self.last_query_vec = None
        self.last_search_was_image = False

        # The VLM caption of the uploaded image (if any) is kept for the whole retrieval
        # session so follow-ups and SQL re-runs still "remember" what the image looked like.
        self.image_analysis = ""

        # Structured, per-query trace: every retrieval step gets logged here AND printed
        self._trace_buffer = []

        # Conversation state (persists across follow-up turns, reset on new search)
        self.history = []  # list of {"role": "user"/"assistant", "content": str}

        # Session log for the whole run
        self.session_log = []

    # Tracing

    def _trace_reset(self):
        self._trace_buffer = []

    def _trace(self, line: str):
        print(line)
        self._trace_buffer.append(line)

    # SQL

    def extract_filter_hints(self, caption: str) -> str:
        """
        EXPERIMENTAL (only used when the researcher ticks the 'use image description to
        filter by metadata' box). Instead of pushing a long free-form caption into the
        SQL prompt, ask the LLM to condense it into at most two short English keywords:
        the object type (print / drawing) and one subject word (portrait, landscape, ...).
        Keeping this narrow limits how much a VLM misreading can over-constrain the query.
        """
        messages = [
            {"role": "system", "content": (
                "You extract search keywords from an image description of an artwork. "
                "Reply with at most TWO lowercase English words separated by a comma: "
                "first the object type (print or drawing), then one main subject "
                "(e.g. portrait, landscape, ship, animal). If unsure about either, leave it out. "
                "Reply with the keywords only."
            )},
            {"role": "user", "content": caption},
        ]
        hints = generate_chat(messages, max_new_tokens=15)
        return hints.strip().splitlines()[0][:60] if hints.strip() else ""

    def generate_sql(self, natural_language_query: str) -> str:
        messages = [
            {"role": "system", "content": SQL_FILTER_SYS},
            {"role": "user", "content": f"Question: {natural_language_query}\nSQL query:"},
        ]
        # Looping is prevented by stopping at the first ';' (not by a repetition penalty, see generate_chat)
        return self._clean_sql(generate_chat(messages, max_new_tokens=250, stop_strings=[";"]))

    def refine_sql(self, previous_sql: str, instruction: str) -> str:
        messages = [
            {"role": "system", "content": REFINE_SQL_SYS},
            {"role": "user", "content": f"Previous SQL: {previous_sql};\nRefinement: {instruction}\nSQL:"},
        ]
        return self._clean_sql(generate_chat(messages, max_new_tokens=300, stop_strings=[";"]))

    @staticmethod
    def _clean_sql(raw_sql: str) -> str:
        clean_sql = re.sub(r"```sql\s*|```|^(sql query:|sql:)\s*", "", raw_sql, flags=re.IGNORECASE).strip()
        # Keep only the first statement (small models sometimes continue after the query)
        return clean_sql.split(";")[0].strip()

    def run_sql(self, sql_text: str):
        """
        Returns (rowids, sql_for_ids); sql_for_ids is the rowid-only query, reused as a subquery
        to load the full result set
        """

        sql_text = sql_text.split(";")[0].strip()
        lowered = sql_text.lower()
        if DANGEROUS_PATTERN.search(lowered):
            raise ValueError("Query rejected: contains a disallowed keyword.")
        if not lowered.startswith("select"):
            raise ValueError("Query rejected: only SELECT statements are allowed.")
        if not re.search(r"(?i)\bFROM\b", sql_text):
            raise ValueError("Query rejected: no FROM clause.")

        # Whatever the query selects, only rowids are needed (?s = DOTALL, the SELECT list may span lines)
        sql_for_ids = re.sub(r"(?is)^SELECT\s+.*?\s+FROM\s+\S+", f"SELECT rowid FROM {self.table_name}",
                             sql_text, count=1)

        cursor = self.db_conn.cursor()
        cursor.execute(sql_for_ids)
        return [int(r[0]) for r in cursor.fetchall() if r[0] is not None], sql_for_ids

    #  Vector search

    def encode_query(self, text: Optional[str] = None, image: Optional[Image.Image] = None) -> Optional[np.ndarray]:
        """
        Embeds the query with CLIP into one L2-normalized (1, dim) vector; image takes priority over text
        """

        if image is not None:
            vec = self.embed_model.encode(image)
        elif text:
            vec = self.embed_model.encode(text)
        else:
            return None
        vec = np.ascontiguousarray(vec, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        return vec

    def vector_search(self, query_vec: np.ndarray, threshold: float) -> list:
        """
        Returns (rowID, score) pairs from the whole index, above the threshold
        """

        scores, indices = self.faiss_index.search(query_vec, VECTOR_SEARCH_K)
        return [(int(self.row_ids[idx]), float(score)) for score, idx in zip(scores[0], indices[0])
                if idx != -1 and score >= threshold]

    def score_candidates(self, query_vec: np.ndarray, candidate_ids: list) -> list:
        """
        Cosine similarity between the query and specific rows (index vectors are already L2-normalized)
        Used to rank SQL hits, instead of intersecting them with CLIP's global top-k
        Returns (rowID, score) pairs sorted by score; rows without an indexed image are left out
        """

        pairs = [(rid, self.rowid_to_pos[rid]) for rid in candidate_ids if rid in self.rowid_to_pos]
        if not pairs:
            return []
        vecs = self.faiss_index.reconstruct_batch(np.array([pos for _, pos in pairs], dtype="int64"))
        sims = vecs @ query_vec.ravel()
        return sorted(zip([rid for rid, _ in pairs], sims.tolist()), key=lambda x: x[1], reverse=True)

    # Results: ranking, sample, profile, context

    def load_records(self, sql_for_ids: str, extra_ids: list) -> pd.DataFrame:
        """
        ALL records of the result set in one DataFrame: the SQL matches (the SQL query is used as a
        subquery, so their number doesn't matter) plus extra rowids from the vector search
        """

        parts = []
        if sql_for_ids:
            parts.append(pd.read_sql_query(
                f"SELECT rowid AS rowid, * FROM {self.table_name} WHERE rowid IN ({sql_for_ids})", self.db_conn))
        if extra_ids:
            placeholders = ",".join("?" for _ in extra_ids)
            parts.append(pd.read_sql_query(
                f"SELECT rowid AS rowid, * FROM {self.table_name} WHERE rowid IN ({placeholders})",
                self.db_conn, params=extra_ids))
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts).drop_duplicates("rowid")
        df["year"] = pd.to_numeric(df[DATE_COLUMN], errors="coerce") # numeric copy for sorting and the profile
        return df.sort_values(["year", "rowid"]).reset_index(drop=True)

    @staticmethod
    def chronological_sample(results: pd.DataFrame, k: int = RETRIEVAL_TOP_K) -> list:
        """
        Evenly spaced picks from ALL matches in chronological order, so the sample mirrors the date
        distribution of the full result set
        """

        ordered = results["rowid"].tolist() # load_records already sorted them by year
        if len(ordered) <= k:
            return ordered
        step = len(ordered) / k
        return [ordered[int((i + 0.5) * step)] for i in range(k)] # the middle of each of the k slices

    def build_profile(self, results: pd.DataFrame) -> str:
        """
        Exact facts about ALL matches (not just the sample). Shown to the researcher and given to the
        answer LLM, so statements about the whole result set don't depend on the sample
        """

        n = len(results)
        lines = [f"Total matches: {n}",
                 f"Distinct titles: {results[TITLE_COLUMN].nunique()} (records with the same title can be several "
                 f"impressions, states or parts of one print)"]

        lo, hi = PLAUSIBLE_YEARS
        years = results["year"]
        plausible = years.between(lo, hi)
        if plausible.any():
            first, last = int(years[plausible].min()), int(years[plausible].max())
            bin_size = 10 if last - first <= 100 else 25 if last - first <= 250 else 50
            periods = (years[plausible] // bin_size * bin_size).astype(int).value_counts().sort_index()
            lines.append(f"Creation dates: {first}-{last}")
            lines.append(f"Per {bin_size} years: " + ", ".join(
                f"{p}-{p + bin_size - 1}: {c}" for p, c in periods.items()))
        n_odd = int((~plausible).sum())
        if n_odd:
            odd = results.loc[~plausible, [OBJECT_NUMBER_COLUMN, DATE_COLUMN]].head(10)
            lines.append(f"Missing or implausible creation dates (outside {lo}-{hi}, likely data errors): {n_odd} - "
                         f"e.g. " + "; ".join(f"{o}: {d!r}" for o, d in odd.itertuples(index=False)))

        lines.append("Object types: " + "; ".join(
            f"{t} ({c})" for t, c in results[TYPE_COLUMN].value_counts().head(10).items()))
        lines.append("Most frequent creators (top 10): " + "; ".join(
            f"{t} ({c})" for t, c in results[CREATOR_COLUMN].value_counts().head(10).items()))
        n_image = int(results[IMAGE_COLUMN].astype(str).str.startswith("http").sum())
        n_indexed = int(results["rowid"].isin(list(self.rowid_to_pos)).sum())
        lines.append(f"With an image: {n_image} of {n}")
        lines.append(f"In the visual (CLIP) index: {n_indexed} of {n} "
                     f"(only these can be ranked or found by visual similarity)")
        return "\n".join(lines)

    def retrieve(self, sql_ids: list, sql_for_ids: str, vector_hits: list, image_mode: bool,
                 query_vec: Optional[np.ndarray]):
        """
        Builds the full result set, chooses the sample the LLM reads and prepares its context.
        SQL returns rows in rowid order, which says nothing about relevance, so:
        - Text mode, SQL found rows: SQL filters. CLIP ranks them if there is a query vector;
          otherwise the sample is a chronological spread over ALL matches
        - Text mode, SQL found nothing: CLIP's global hits
        - Image mode: SQL hits (ranked by similarity to the image) and CLIP's global hits, fused with RRF
        """

        if sql_ids and not image_mode:
            vector_hits = [] # in text mode CLIP only ranks SQL hits, it doesn't add records
        vector_ids = [rid for rid, _ in vector_hits]
        scores = dict(vector_hits)

        if sql_ids and query_vec is not None:
            ranked = self.score_candidates(query_vec, sql_ids)
            scores.update(ranked)
            ranked_ids = [rid for rid, _ in ranked]
            sql_ids = ranked_ids + [rid for rid in sql_ids if rid not in set(ranked_ids)]
            self._trace(f"[Rank] Re-ranked {len(ranked)} SQL hit(s) by CLIP similarity to the query; "
                        f"{len(sql_ids) - len(ranked)} without an indexed image kept at the end.")

        results = self.load_records(sql_for_ids if sql_ids else "", vector_ids)
        if results.empty:
            self.results = self.context_df = pd.DataFrame()
            self.context_str, self.images, self.profile, self.ranking_note = "", [], "", ""
            return

        # The sample: which records the answer LLM reads in detail, and how they were chosen
        if image_mode or not sql_ids:
            sample_ids = reciprocal_rank_fusion(sql_ids, vector_ids)
            if image_mode and sql_ids:
                self.ranking_note = "metadata matches and visually similar artworks, fused by similarity to the uploaded image"
            elif image_mode:
                self.ranking_note = "visual similarity to the uploaded image only"
            else:
                self.ranking_note = "visual similarity only (the SQL query found nothing)"
        elif query_vec is not None:
            sample_ids = sql_ids[:RETRIEVAL_TOP_K]
            self.ranking_note = "the metadata matches most similar to the question (CLIP)"
        else:
            sample_ids = self.chronological_sample(results)
            self.ranking_note = "records evenly spread over the chronological order of all matches"
        self._trace(f"[Rank] Sample: {self.ranking_note}.")
        self._trace(f"[Rank] Final selection ({len(sample_ids)} of {len(results)} artwork(s)): {sample_ids}")

        # Provenance and position in the context, for both tables
        position = {rid: n for n, rid in enumerate(sample_ids, start=1)}
        sql_set, vector_set = set(sql_ids), set(vector_ids)
        results.insert(0, "Artwork #", results["rowid"].map(lambda r: position.get(r, "")))
        results.insert(1, "found_via", results["rowid"].map(
            lambda r: "+".join(s for s, hit in (("SQL", r in sql_set), ("Vector", r in vector_set)) if hit)))
        results.insert(2, "vector_similarity", results["rowid"].map(lambda r: round(scores[r], 3) if r in scores else ""))

        self.profile = self.build_profile(results)
        self._trace("[Profile] Exact facts about all matches:\n" + self.profile)
        self.results = results.drop(columns="year")
        self.context_df = (self.results[self.results["rowid"].isin(sample_ids)]
                           .sort_values("Artwork #").reset_index(drop=True))

        # Context for the LLM (object number as identifier) and captioned images for the gallery
        context_parts, self.images = [], []
        for _, row in self.context_df.iterrows():
            n, obj = row["Artwork #"], row[OBJECT_NUMBER_COLUMN]
            details = " | ".join(f"{label}: {row[col]}" for col, label in CONTEXT_FIELDS if _is_filled(row[col]))
            context_parts.append(f"[Artwork #{n} | {obj}] {details}")
            if str(row[IMAGE_COLUMN]).startswith("http"):
                self.images.append((row[IMAGE_COLUMN], f"Artwork #{n} ({obj}) - {row[TITLE_COLUMN]}"))
        self.context_str = "\n\n".join(context_parts)

    # Synthesis

    def synthesize(self, user_question: str) -> str:
        """
        Returns the answer for the UI (with a scope line and sources); the history keeps the plain answer
        """

        system_content = SYNTHESIS_SYS_TEMPLATE.format(context=self.context_str)
        n_read, n_total = len(self.context_df), len(self.results)
        system_content += (
            "\nExact facts about ALL matches of the search (computed by the database; use these for any "
            f"statement about the whole result set):\n{self.profile}\n"
        )
        if n_total > n_read:
            system_content += (f"\nNote: {n_total} artworks matched, but only the {n_read} above are in your context "
                               f"({self.ranking_note}). Do not generalize from them to the full set of matches.\n")
        else:
            system_content += "\nNote: the context above contains ALL artworks that matched the search.\n"
        if self.image_analysis:
            system_content += (
                "\nThe researcher also uploaded an image. An automatic (small-model) description "
                "of it follows; treat it as a rough, possibly inaccurate aid, not as fact:\n"
                f"{self.image_analysis}\n"
            )
        system_content += ("\nThis is a FOLLOW-UP question.\n" if self.history else
                           "\nThis is a NEW question: use the Answer / Observations / Limits structure.\n")

        messages = [{"role": "system", "content": system_content}] + self.history
        messages.append({"role": "user", "content": user_question})

        # Tweakable parameters (lower temperature keeps the answer closer to the context; repetition_penalty
        # is kept mild because it also penalizes titles and names copied from the context)
        answer = generate_chat(messages, max_new_tokens=900, temperature=0.3, top_p=0.9, top_k=50,
                               repetition_penalty=1.05)

        self.history.append({"role": "user", "content": user_question})
        self.history.append({"role": "assistant", "content": answer})
        return self.attach_sources(answer)

    def attach_sources(self, answer: str) -> str:
        """
        Adds, written by the code and not by the LLM: a scope line (what the answer is based on), the cited
        records from the database (so every citation can be checked), and warnings for invalid or missing citations
        """

        n_read, n_total = len(self.context_df), len(self.results)
        scope = (f"[Based on all {n_total} matching records.]" if n_read >= n_total else
                 f"[Based on {n_read} of {n_total} matching records: {self.ranking_note}. "
                 f"Exact figures over all matches are in the profile.]")

        cited = sorted({int(n) for n in re.findall(r"Artwork #(\d+)", answer)})
        lines = []
        for n in cited:
            match = self.context_df[self.context_df["Artwork #"] == n]
            if match.empty:
                lines.append(f"WARNING: the answer cites Artwork #{n}, which is not in the context.")
                continue
            row = match.iloc[0]
            lines.append(f"Artwork #{n}: " + " | ".join(str(row[c]) for c in (
                OBJECT_NUMBER_COLUMN, TITLE_COLUMN, CREATOR_COLUMN, DATE_COLUMN, HANDLE_COLUMN) if _is_filled(row[c])))
        if not cited:
            lines.append("WARNING: the answer cites no specific records; its claims are not grounded in the data.")
        self._trace("[Check] " + ("; ".join(l for l in lines if l.startswith("WARNING")) or
                                  "All cited artwork numbers exist in the context."))
        return (scope + "\n\n" + answer + "\n\n---\nSources (from the database, not generated by the model):\n"
                + "\n".join(lines))

    # Top-level orchestration

    def new_search(self, user_input: str, user_image: Optional[Image.Image] = None,
                   include_vector_for_text: bool = False, use_caption_for_sql: bool = False):
        """
        Full pipeline: SQL + optional vector search -> results, sample and profile -> reset conversation -> synthesize

        - SQL handles what the researcher TYPED (plus, optionally, condensed image hints)

        - CLIP finds visually similar artworks from an uploaded image (always runs in image
          mode; opt-in for text-only queries via `include_vector_for_text`), and ranks the SQL hits

        - The VLM caption is a soft input for the answer LLM and a visible aid for the
          researcher. It is NOT pushed into SQL unless `use_caption_for_sql` is True, because
          a small VLM's guesses become hard WHERE filters that silently exclude good matches

        """

        self._trace_reset()
        image_mode = user_image is not None
        self._trace(f"[New search] query={user_input!r}, image_uploaded={image_mode}, "
                    f"include_vector_for_text={include_vector_for_text}, use_caption_for_sql={use_caption_for_sql}")

        self.image_analysis = analyze_image(user_image) if image_mode else ""
        if image_mode:
            self._trace(f"[VLM] Description: {self.image_analysis or '(none: VLM disabled or failed, see console)'}")

        sql_question = (user_input or "").strip()
        if self.image_analysis and use_caption_for_sql:
            hints = self.extract_filter_hints(self.image_analysis)
            self._trace(f"[SQL] Experimental: condensed image hints for SQL = {hints!r}")
            sql_question = f"{sql_question} {hints}".strip()

        sql, sql_ids, sql_for_ids = "", [], ""
        if sql_question:
            try:
                sql = self.generate_sql(sql_question)
                self._trace(f"[SQL] Generated SQL: {sql}")
                sql_ids, sql_for_ids = self.run_sql(sql)
                self._trace(f"[SQL] Matched {len(sql_ids)} row(s): {_preview_ids(sql_ids)}")
            except Exception as e:
                self._trace(f"[SQL Error] {e}")
        else:
            self._trace("[SQL] Skipped (no typed question: image-only search relies on CLIP).")

        # CLIP: always for an uploaded image; for text only when requested
        query_vec, vector_hits = None, []
        if not (image_mode or include_vector_for_text):
            self._trace("[Vector] Skipped (text-only query, 'include visual similarity' not checked).")
        elif self.faiss_index.ntotal == 0:
            self._trace(f"[Vector] {INDEX_STATUS}")
        else:
            query_vec = self.encode_query(text=user_input, image=user_image)
            # Global search in image mode, or as a fallback when the text query found nothing
            if image_mode or not sql_ids:
                threshold = VECTOR_SIMILARITY_THRESHOLD_IMAGE if image_mode else VECTOR_SIMILARITY_THRESHOLD_TEXT
                vector_hits = self.vector_search(query_vec, threshold)
                self._trace(f"[Vector] {len(vector_hits)} candidate(s) above threshold {threshold}: " +
                            ", ".join(f"{rid}:{score:.3f}" for rid, score in vector_hits))

        self.retrieve(sql_ids, sql_for_ids, vector_hits, image_mode, query_vec)
        self.last_sql, self.last_sql_ids = sql, sql_ids
        self.last_vector_hits, self.last_query_vec, self.last_search_was_image = vector_hits, query_vec, image_mode
        self.last_question = (user_input or "").strip() or (
            "Describe the retrieved artworks and how they relate to the uploaded image." if image_mode
            else "Describe and compare the retrieved artworks.")
        return self._finish("new_search", user_input, self.last_question, sql)

    def refine_search(self, instruction: str):
        """
        Narrow (or adjust) the CURRENT result set: the LLM edits the last SQL query instead of writing a new
        one, so earlier conditions (e.g. the creator) are kept. The vector half of the last search is reused
        """

        if not self.last_sql:
            return self.new_search(instruction)

        self._trace_reset()
        self._trace(f"[Refine] instruction={instruction!r}")
        self._trace(f"[Refine] Previous SQL: {self.last_sql}")
        refined_sql = ""
        try:
            refined_sql = self.refine_sql(self.last_sql, instruction)
            self._trace(f"[Refine] Refined SQL: {refined_sql}")
            sql_ids, sql_for_ids = self.run_sql(refined_sql)
            self._trace(f"[Refine] Matched {len(sql_ids)} row(s): {_preview_ids(sql_ids)}")
        except Exception as e:
            return self._error(f"Refined SQL rejected: {e}", refined_sql or self.last_sql)

        # A refinement should narrow the result set; if it doesn't, the researcher needs to know
        previous = set(self.last_sql_ids)
        n_outside = sum(1 for rid in sql_ids if rid not in previous)
        warning = (f"NOTE: {n_outside} of the {len(sql_ids)} matches were NOT in the previous result set, so this "
                   f"refinement widened or changed the search instead of narrowing it. Check the SQL."
                   if n_outside else "")
        self._trace(f"[Refine] {warning or f'All {len(sql_ids)} matches lie within the previous {len(previous)}.'}")

        self.retrieve(sql_ids, sql_for_ids, self.last_vector_hits, self.last_search_was_image, self.last_query_vec)
        self.last_sql, self.last_sql_ids = refined_sql, sql_ids
        self.last_question = f"{self.last_question} (refined: {instruction})"
        return self._finish("refine", instruction, self.last_question, refined_sql, warning)

    def follow_up(self, user_question: str):
        """
        Reason over the currently retrieved context without re-running retrieval
        """

        if self.context_df.empty:
            return self.new_search(user_question)

        self._trace_reset()
        self._trace(f"[Follow-up] query={user_question!r}")
        self._trace(f"[Follow-up] No retrieval re-run. Reusing the {len(self.context_df)} artwork(s) in the context; "
                    f"{len(self.history) // 2} prior turn(s)")

        answer = self.synthesize(user_question)
        self._log("follow_up", user_question, self.last_sql, answer)
        return self._result(answer, self.last_sql)

    def rerun_edited_sql(self, edited_sql: str):
        """
        Re-run a researcher-edited SQL string, keep prior vector hits and query vector, refresh results, resynthesize
        """

        self._trace_reset()
        self._trace(f"[Re-run SQL] Researcher-edited query: {edited_sql}")
        try:
            sql_ids, sql_for_ids = self.run_sql(edited_sql)
            self._trace(f"[Re-run SQL] Matched {len(sql_ids)} row(s): {_preview_ids(sql_ids)}")
        except Exception as e:
            return self._error(f"SQL rejected: {e}", edited_sql)

        self.retrieve(sql_ids, sql_for_ids, self.last_vector_hits, self.last_search_was_image, self.last_query_vec)
        self.last_sql, self.last_sql_ids = edited_sql, sql_ids
        return self._finish("rerun_sql", "(edited SQL)", "Describe and compare the retrieved artworks.", edited_sql)

    def _finish(self, mode: str, log_query: str, question: str, sql: str, warning: str = ""):
        """
        Common end of every retrieval: reset the conversation, answer (or report no matches), log
        """

        self.history = []
        answer = self.synthesize(question) if not self.context_df.empty else "No artworks matched this query."
        if warning:
            answer = f"{warning}\n\n{answer}"
        self._log(mode, log_query, sql, answer)
        return self._result(answer, sql)

    def _result(self, answer: str, sql: str) -> dict:
        return {"answer": answer, "images": self.images, "sql": sql, "table": self.context_df,
                "trace": "\n".join(self._trace_buffer), "vlm_description": self.image_analysis,
                "profile": self.profile, "all_matches": self.results.head(MAX_RESULT_ROWS_DISPLAY),
                "n_matches": len(self.results)}

    def _error(self, message: str, sql: str) -> dict:
        self._trace(f"[Error] {message}")
        return {"answer": message, "images": [], "sql": sql, "table": pd.DataFrame(),
                "trace": "\n".join(self._trace_buffer), "vlm_description": "", "profile": "",
                "all_matches": pd.DataFrame(), "n_matches": 0}

    def _log(self, mode: str, query: str, sql: str, answer: str):
        # Which records the answer was based on, and how they were chosen, are logged with every answer
        self.session_log.append({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "query": query,
            "sql": sql,
            "answer": answer,
            "total_matches": len(self.results),
            "ranking_note": self.ranking_note,
            "context_records": ";".join(self.context_df[OBJECT_NUMBER_COLUMN].astype(str)) if not self.context_df.empty else "",
            "llm": LLM_MODEL_NAME,
        })

    def export_log_csv(self) -> str:
        out_path = str(CURRENT_DIR / "session_log.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "mode", "query", "sql", "answer", "total_matches",
                                                   "ranking_note", "context_records", "llm"])
            writer.writeheader()
            writer.writerows(self.session_log)
        return out_path

    def export_results_csv(self) -> Optional[str]:
        """
        Exports ALL matches of the current search (no display limit)
        """

        if self.results.empty:
            return None
        out_path = str(CURRENT_DIR / "search_results.csv")
        self.results.to_csv(out_path, index=False, encoding="utf-8")
        return out_path


engine = ResearchEngine(conn, MAIN_TABLE, faiss_index, faiss_row_ids, embedding_model)


# 8. GRADIO FRONTEND INTERFACE


def _outputs(result: dict) -> tuple:
    # The two table titles state the actual numbers of each search, so the boxes can't be misread
    n_all, n_context = result["n_matches"], len(result["table"])
    context_table = gr.Dataframe(value=result["table"],
                                 label=f"Records in the model's context: {n_context} of {n_all} matches (what the AI reads)")
    all_table = gr.Dataframe(value=result["all_matches"], label=f"All matches: {n_all} records found by the search" + (
        f" (first {MAX_RESULT_ROWS_DISPLAY} shown here; the CSV export has all)" if n_all > MAX_RESULT_ROWS_DISPLAY else ""))
    return (result["answer"], result["images"], result["sql"], context_table, pd.DataFrame(engine.session_log),
            result["trace"], result["vlm_description"], result["profile"], all_table)


def handle_query(user_query: str, uploaded_image: Optional[Image.Image], mode: str,
                 include_vector_for_text: bool, use_caption_for_sql: bool):
    if not user_query or not user_query.strip():
        if mode != "New search" or uploaded_image is None:
            return ("Please provide a question or upload an image.", [], "", pd.DataFrame(),
                    pd.DataFrame(engine.session_log), "", "", "", pd.DataFrame())
    try:
        image = uploaded_image.convert("RGB") if uploaded_image is not None else None
        if mode == "New search":
            result = engine.new_search(user_query, image, include_vector_for_text=include_vector_for_text,
                                       use_caption_for_sql=use_caption_for_sql)
        elif mode == "Refine current search":
            result = engine.refine_search(user_query)
        else:
            result = engine.follow_up(user_query)
        return _outputs(result)
    except Exception as e:
        return _outputs(engine._error(f"Error: {e}", ""))
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def handle_rerun_sql(edited_sql: str):
    try:
        return _outputs(engine.rerun_edited_sql(edited_sql))
    except Exception as e:
        return _outputs(engine._error(f"Error: {e}", edited_sql))


with gr.Blocks(title="Rijksmuseum Research Assistant") as demo:
    gr.Markdown(
        f"""
        # Rijksmuseum Graphic Arts Research Assistant
        Text-to-SQL + multimodal vector search over the early modern prints and
        drawings collection. The generated SQL and retrieved records are shown below so
        you can inspect and correct retrieval, not just the final answer — including
        whether each row came from SQL, visual similarity, or both.

        *{INDEX_STATUS}*

        The model reads a **sample** of the matches in detail; exact figures about **all** matches
        are in the profile, and the full result set is in the "All matches" table and CSV export.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            mode_toggle = gr.Radio(
                ["New search", "Refine current search", "Follow-up (reason over current results)"],
                value="New search",
                label="Mode",
                info=("Refine: narrows the full result set of the current search (e.g. 'only those after 1575'). "
                      "Follow-up: reasons only over the records in the model's context, without searching again."),
            )
            user_input = gr.Textbox(label="Question", lines=3,
                                    placeholder="e.g. Which prints were produced by Cornelis Cort?")
            image_input = gr.Image(label="Upload image (optional, new search only)", type="pil")
            vector_for_text_checkbox = gr.Checkbox(
                label="Also use visual similarity search (CLIP) for this text query",
                value=False,
                info=(
                    "Off by default: CLIP text-image matching only works well for short, topic-heavy phrasing "
                    "(e.g. 'farm animals', 'landscapes'). It only re-ranks the SQL matches (or searches on its own "
                    "if SQL finds nothing). Always runs automatically when you upload an image."
                ),
            )
            caption_for_sql_checkbox = gr.Checkbox(
                label="Use image description to filter by metadata (experimental)",
                value=False,
                info=(
                    "Off by default. The small vision model can misjudge medium or period, and as an SQL "
                    "filter those mistakes would silently exclude good matches. When ticked, only "
                    "two condensed keywords (object type, subject) are added to your typed question."
                ),
            )
            submit_btn = gr.Button("Submit", variant="primary")

            gr.Markdown("**Generated SQL** (editable — edit and re-run to steer retrieval directly)")
            sql_box = gr.Textbox(label="SQL", lines=4)
            rerun_btn = gr.Button("Re-run edited SQL")

            export_results_btn = gr.Button("Export all matches (CSV)")
            export_results_file = gr.File(label="Search results download")
            export_btn = gr.Button("Export session log (CSV)")
            export_file = gr.File(label="Session log download")

        with gr.Column(scale=2):
            output_answer = gr.Textbox(label="Answer (with sources from the database)", lines=12, interactive=False)
            profile_box = gr.Textbox(label="Result set profile (exact, computed over ALL matches)", lines=6,
                                     interactive=False)
            output_gallery = gr.Gallery(label="Matched artwork images (in the model's context)", columns=3, height=300)
            output_table = gr.Dataframe(label="Records in the model's context", wrap=True)
            all_matches_table = gr.Dataframe(label="All matches", wrap=True)
            vlm_box = gr.Textbox(label="VLM description of uploaded image", lines=3, interactive=False)
            with gr.Accordion("Retrieval trace (debug)", open=False):
                trace_box = gr.Textbox(label="Step-by-step trace", lines=16, interactive=False)
            session_log_table = gr.Dataframe(label="Session log", wrap=True)

    all_outputs = [output_answer, output_gallery, sql_box, output_table, session_log_table, trace_box, vlm_box,
                   profile_box, all_matches_table]
    submit_btn.click(
        fn=handle_query,
        inputs=[user_input, image_input, mode_toggle, vector_for_text_checkbox, caption_for_sql_checkbox],
        outputs=all_outputs,
    )
    rerun_btn.click(fn=handle_rerun_sql, inputs=[sql_box], outputs=all_outputs)
    export_results_btn.click(fn=engine.export_results_csv, outputs=[export_results_file])
    export_btn.click(fn=engine.export_log_csv, outputs=[export_file])

if __name__ == "__main__":
    # share=True for Colab (has no accessible localhost)
    demo.queue().launch(share=False, debug=False)