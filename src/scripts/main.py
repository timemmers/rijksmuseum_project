"""
Rijksmuseum Graphics Arts AI Assistant - A Text-to-SQL and Multimodal Vector Search AI system with Gradio UI
---
python -m pip install torch transformers faiss-cpu accelerate numpy requests gradio pillow sentence-transformers pandas bitsandbytes
"""

from pathlib import Path
import re
import sqlite3
import io
from typing import Optional
import csv

import requests
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
VLM_MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"
CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

CURRENT_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
DATABASE_PATH = CURRENT_DIR.parent / "preprocessing" / "rma_artworks"

MAX_INDEX_IMAGES = 1500

RETRIEVAL_TOP_K = 7
VECTOR_SEARCH_K = 15

VECTOR_SIMILARITY_THRESHOLD = 0.24 # Minimum cosine similarity to drop weak matches before RRF pollution


# 1. DATABASE SETUP & SCHEMA EXTRACTION

def connect_and_get_schema(db_path: Path):
    """
    Connects to SQLite database, detects existing tables,
    and constructs a readable schema string for Text-to-SQL prompting
    """

    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]

    main_table = tables[0]
    db_schema = {}

    for table_name in tables:
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        columns = [f"{col_info[1]} ({col_info[2]})" for col_info in cursor.fetchall()]
        db_schema[table_name] = columns

    schema_str = "".join(f"Table: {t}\nColumns: {', '.join(cols)}\n\n" for t, cols in db_schema.items())

    print(f"[Database] Connected to '{db_path}'. Primary table detected: '{main_table}'")
    return conn, main_table, schema_str

conn, MAIN_TABLE, SCHEMA_STR = connect_and_get_schema(DATABASE_PATH)

def _find_image_column(db_conn: sqlite3.Connection, table_name: str) -> Optional[str]:
    cursor = db_conn.cursor()
    cursor.execute(f"PRAGMA table_info('{table_name}');")
    columns = [c[1] for c in cursor.fetchall()]
    return next((c for c in columns if "image" in c.lower() or "url" in c.lower()), None)

IMAGE_COLUMN = _find_image_column(conn, MAIN_TABLE)


# 2. SYSTEM PROMPTS

SQL_FILTER_SYS = f"""You are an expert AI assistant that translates natural language questions into executable SQLite SQL queries.

Database schema:
{SCHEMA_STR}

CRITICAL RULES:
1. Scope: the database only contains early modern prints and drawings. Filter using "objectType[1]" and "objectCreationDate[1]" where relevant.
2. Use EXACT column names with brackets, e.g. "objectType[1]", "objectCreator[1]", "objectCreationDate[1]".
3. The database uses Dutch terms. Translate English search terms into Dutch for LIKE clauses.
   Examples: 'print' -> 'prent', 'drawing' -> 'tekening', 'landscape' -> 'landschap', 'portrait' -> 'portret'.
4. Creator names are stored as 'Lastname, Firstname' (e.g. 'Cort, Cornelis'). Never match a full name as one string.
   Instead: WHERE "objectCreator[1]" LIKE '%Cornelis%' AND "objectCreator[1]" LIKE '%Cort%'
5. If the question includes a visual description (from an uploaded image) rather than explicit search terms,
   infer plausible object type / subject / period filters from it the same way you would from a text question.

FEW-SHOT EXAMPLES:

Question: Which prints were produced by Cornelis Cort?
SQL: SELECT rowid FROM {MAIN_TABLE} WHERE "objectType[1]" LIKE '%prent%' AND "objectCreator[1]" LIKE '%Cornelis%' AND "objectCreator[1]" LIKE '%Cort%' AND "objectCreationDate[1]" BETWEEN '1450' AND '1850'

Question: Show me drawings of landscapes from the 16th century
SQL: SELECT rowid FROM {MAIN_TABLE} WHERE "objectType[1]" LIKE '%tekening%' AND "objectTitle[1]" LIKE '%landschap%' AND "objectCreationDate[1]" BETWEEN '1501' AND '1600'

Question: Show me 17th century portraits
SQL: SELECT rowid FROM {MAIN_TABLE} WHERE "objectType[1]" LIKE '%prent%' AND "objectTitle[1]" LIKE '%portret%' AND "objectCreationDate[1]" BETWEEN '1601' AND '1700'

Return ONLY the raw SQL query, nothing else."""

SYNTHESIS_SYS_TEMPLATE = """You are an expert art historian assisting a researcher in exploring the Rijksmuseum's
early modern print and drawing collection.

You are given a set of retrieved artwork records as context. Use ONLY this context and the
conversation so far — do not invent facts that are not present in it.

Guidelines:
- Ground every claim in a specific artwork from the context (reference it as "Artwork #N").
- You do NOT need to describe every artwork in a fixed order. Prioritize what is analytically
  interesting: patterns, contrasts, outliers, likely attributions, stylistic or thematic links.
- If the researcher asks a follow-up question, answer it directly using the same context and
  the conversation history — you don't need to re-summarize everything from scratch.
- If the context doesn't contain enough information to answer confidently, say so explicitly
  rather than guessing.
- Write for a researcher: precise, willing to flag uncertainty, comfortable making a reasoned
  comparative judgment rather than only describing.

Database Context (retrieved artworks for this session):
{context}
"""


# 3. LLM SETUP

print(f"[LLM] Loading {LLM_MODEL_NAME}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
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
                  top_p: float = 1.0, top_k: int = 50, repetition_penalty: float = 1.1) -> str:
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
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_cfg.temperature = temperature
        gen_cfg.top_p = top_p
        gen_cfg.top_k = top_k

    output_ids = llm_model.generate(input_ids, attention_mask=attention_mask, generation_config=gen_cfg)
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


# 4. VLM SETUP

vlm_processor = None
vlm_model = None
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

print(f"[Embeddings] Loading CLIP model ({CLIP_MODEL_NAME})...")
embedding_model = SentenceTransformer(CLIP_MODEL_NAME)

def build_or_load_faiss_index(db_conn, table_name, image_col, embed_model, max_images=MAX_INDEX_IMAGES):
    """
    Checks if FAISS vector index exists. If missing, automatically extracts image URLs from early modern prints
    and drawings (up to max_images), downloads images, and builds the FAISS index using GPU if available
    """

    index_file = CURRENT_DIR / "artworks.index"
    row_ids_file = CURRENT_DIR / "row_ids.npy"

    if index_file.exists() and row_ids_file.exists():
        print(f"[FAISS] Found existing index files: '{index_file.name}' and '{row_ids_file.name}'. Loading...")
        return faiss.read_index(str(index_file)), np.load(str(row_ids_file))

    if image_col is None:
        print("[FAISS] No image column available - return empty index...")
        return faiss.IndexFlatIP(512), np.array([], dtype=int)

    print(f"[FAISS] Starting automatic image embedding for early modern prints/drawings (Limited to max {max_images} images for testing)...")
    cursor = db_conn.cursor()
    query = f"""
            SELECT rowid, {image_col} 
            FROM {table_name} 
            WHERE ("objectType[1]" LIKE '%prent%' OR "objectType[1]" LIKE '%tekening%')
              AND "objectCreationDate[1]" BETWEEN '1450' AND '1850'
              AND {image_col} IS NOT NULL AND {image_col} != '<null>' AND {image_col} != '' 
            LIMIT {max_images}
        """
    cursor.execute(query)
    records = cursor.fetchall()
    print(f"[FAISS] Found {len(records)} image records to process.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model.to(device)
    print(f"[FAISS] Running embedding model on device: {device.upper()}")

    embeddings, valid_row_ids = [], []
    headers = {'User-Agent': 'Mozilla/5.0'}

    for i, (row_id, img_url) in enumerate(records):
        try:
            if isinstance(img_url, str) and img_url.startswith("http"):
                res = requests.get(img_url, timeout=5, headers=headers)
                if res.status_code == 200:
                    img = Image.open(io.BytesIO(res.content)).convert("RGB")
                    embeddings.append(embed_model.encode(img, convert_to_numpy=True))
                    valid_row_ids.append(row_id)
        except Exception as err:
            pass
        if (i + 1) % 100 == 0:
            print(f"  ... processed {i + 1}/{len(records)}")

    if not embeddings:
        print("[FAISS Warning] No images embedded. Returning empty index.")
        return faiss.IndexFlatIP(512), np.array([], dtype=int)

    embeddings_np = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings_np)
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    row_ids_np = np.array(valid_row_ids, dtype=int)

    faiss.write_index(index, str(index_file))
    np.save(str(row_ids_file), row_ids_np)
    print(f"[FAISS] Index built with {len(valid_row_ids)} vectors.")
    return index, row_ids_np


faiss_index, faiss_row_ids = build_or_load_faiss_index(conn, MAIN_TABLE, IMAGE_COLUMN, embedding_model)


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

class ResearchEngine:
    """
    Holds retrieval state (current context) and conversation history separately,
    so a follow-up question can reason over the same retrieved artworks without
    re-running SQL/vector search, and a "new search" explicitly resets both
    """

    def __init__(self, db_conn, table_name, image_col, faiss_idx, row_ids_arr, embed_model):
        self.db_conn = db_conn
        self.table_name = table_name
        self.image_col = image_col
        self.faiss_index = faiss_idx
        self.row_ids = row_ids_arr
        self.embed_model = embed_model

        # Retrieval state (persists across follow-up turns)
        self.current_context_str = ""
        self.current_hybrid_ids = []
        self.current_image_urls = []
        self.last_sql_query = ""
        self.last_vector_ids = []
        self.last_vector_scores = {}
        self.current_source_map = {}

        # The VLM caption of the uploaded image (if any) is kept for the whole retrieval
        # session so follow-ups and SQL re-runs still "remember" what the image looked like.
        self.current_image_analysis = ""
        # True if the last search was driven by an uploaded image; controls whether vector
        # hits are unioned with SQL hits (image mode) or restricted to them (text mode).
        self.last_search_was_image = False

        # Structured, per-query trace: every retrieval step gets logged here AND printed
        self._trace_buffer = []
        self.last_trace = ""

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

    def _trace_finalize(self) -> str:
        self.last_trace = "\n".join(self._trace_buffer)
        return self.last_trace

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
        hints = generate_chat(messages, max_new_tokens=15, temperature=0.0)
        return hints.strip().splitlines()[0][:60] if hints.strip() else ""

    def generate_sql(self, natural_language_query: str) -> str:
        messages = [
            {"role": "system", "content": SQL_FILTER_SYS},
            {"role": "user", "content": f"Question: {natural_language_query}\nSQL query:"},
        ]
        raw_sql = generate_chat(messages, max_new_tokens=250, temperature=0.0)
        clean_sql = re.sub(r"```sql\s*|```|^(sql query:|sql:)\s*", "", raw_sql, flags=re.IGNORECASE).strip()
        return clean_sql

    def run_sql(self, sql_text: str) -> list:
        lowered = sql_text.lower()
        if any(word in lowered for word in DANGEROUS_KEYWORDS):
            raise ValueError("Query rejected: contains a disallowed keyword.")
        if not lowered.strip().startswith("select"):
            raise ValueError("Query rejected: only SELECT statements are allowed.")

        if re.search(r"(?i)\bFROM\b", sql_text):
            sql_for_ids = re.sub(r"(?i)^SELECT\s+.*?\s+FROM\s+\S+", f"SELECT rowid FROM {self.table_name}", sql_text)
        else:
            sql_for_ids = f"SELECT rowid FROM {self.table_name} WHERE {sql_text}"

        cursor = self.db_conn.cursor()
        cursor.execute(sql_for_ids)
        return [int(r[0]) for r in cursor.fetchall() if r[0] is not None]

    #  Vector search

    def vector_search(self, text: Optional[str] = None, image: Optional[Image.Image] = None,
                      k: int = VECTOR_SEARCH_K, threshold: float = VECTOR_SIMILARITY_THRESHOLD) -> list:
        """
        Returns a list of pairs (rowID, score), filtered by the threshold; image takes priority over text,
        CLIP embeds one query vector per call
        """

        if self.faiss_index.ntotal == 0 or len(self.row_ids) == 0:
            return []
        if image is not None:
            vec = self.embed_model.encode(image).astype("float32")
        elif text:
            vec = self.embed_model.encode(text).astype("float32")
        else:
            return []
        if vec.ndim == 1:
            vec = np.expand_dims(vec, axis=0)
        faiss.normalize_L2(vec)
        scores, indices = self.faiss_index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.row_ids):
                continue
            if score < threshold:
                continue
            results.append((int(self.row_ids[idx]), float(score)))
        return results


    # Context fetching

    def fetch_context(self, hybrid_ids: list, source_map: Optional[dict] = None):
        """
        Returns (context_str, image_urls, dataframe) for the given rowids, in rank order
        source_map = maps rowid so results table can show why each row was retrieved
        """

        if not hybrid_ids:
            return "", [], pd.DataFrame()
        source_map = source_map or {}

        cursor = self.db_conn.cursor()
        placeholders = ",".join("?" for _ in hybrid_ids)
        cursor.execute(f"SELECT rowid, * FROM {self.table_name} WHERE rowid IN ({placeholders})", hybrid_ids)
        raw_rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description][1:]
        row_dict = {r[0]: r[1:] for r in raw_rows}

        context_parts, image_urls, table_rows = [], [], []
        for idx, rid in enumerate(hybrid_ids, start=1):
            if rid not in row_dict:
                continue
            row = row_dict[rid]
            details = [f"{c}: {v}" for c, v in zip(columns, row) if v and str(v).strip()]
            context_parts.append(f"[Artwork #{idx} - ID {rid}] " + " | ".join(details))

            src = source_map.get(rid, {})
            found_via = "+".join(filter(None, ["SQL" if src.get("sql") else "", "Vector" if src.get("vector") else ""])) or "?"
            table_rows.append({
                "Artwork #": idx,
                "rowid": rid,
                "found_via": found_via,
                "vector_similarity": round(src["vector_score"], 3) if src.get("vector_score") is not None else "",
                **{c: v for c, v in zip(columns, row)},
                              })

            for c, v in zip(columns, row):
                if self.image_col and c == self.image_col and isinstance(v, str) and v.startswith("http"):
                    image_urls.append(v)
                    break

        df = pd.DataFrame(table_rows)
        return "\n\n".join(context_parts), image_urls, df

    # Synthesis

    def synthesize(self, user_question: str) -> str:
        system_content = SYNTHESIS_SYS_TEMPLATE.format(context=self.current_context_str)
        if self.current_image_analysis:
            system_content += (
                "\nThe researcher also uploaded an image. An automatic (small-model) description "
                "of it follows; treat it as a rough, possibly inaccurate aid, not as fact:\n"
                f"{self.current_image_analysis}\n"
            )

        messages = [{"role": "system", "content": system_content}] + self.history
        messages.append({"role": "user", "content": user_question})

        answer = generate_chat(messages, max_new_tokens=600, temperature=0.6, top_p=0.9, top_k=50)

        self.history.append({"role": "user", "content": user_question})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    # Top-level orchestration

    def new_search(self, user_input: str, user_image: Optional[Image.Image] = None,
                   include_vector_for_text: bool = False, use_caption_for_sql: bool = False):
        """
        Full pipeline: SQL + optional vector search -> RRF -> reset conversation -> synthesize

        - SQL handles what the researcher TYPED (plus, optionally, condensed image hints)

        - CLIP finds visually similar artworks from an uploaded image (always runs in image
          mode; opt-in for text-only queries via `include_vector_for_text`)

        - The VLM caption is a soft input for the answer LLM and a visible aid for the
          researcher. It is NOT pushed into SQL unless `use_caption_for_sql` is True, because
          a small VLM's guesses become hard WHERE filters that silently exclude good matches

        """

        self._trace_reset()
        image_mode = user_image is not None
        self._trace(f"[New search] query={user_input!r}, image_uploaded={image_mode}, "
                    f"include_vector_for_text={include_vector_for_text}, use_caption_for_sql={use_caption_for_sql}")

        image_analysis = analyze_image(user_image) if image_mode else ""
        self._trace(
            f"[VLM] {'Description: ' + image_analysis if image_analysis else '(no image uploaded, VLM skipped)'}")

        sql_question = (user_input or "").strip()
        if image_analysis and use_caption_for_sql:
            hints = self.extract_filter_hints(image_analysis)
            self._trace(f"[SQL] Experimental: condensed image hints for SQL = {hints!r}")
            if hints:
                sql_question = f"{sql_question} {hints}".strip()
        elif image_analysis:
            self._trace("[SQL] Image description is NOT used for SQL (experimental filter option is off).")
        self._trace(f"[SQL] Question sent to LLM for translation: {sql_question!r}")

        sql_query, sql_ids = "", []
        if sql_question:
            try:
                sql_query = self.generate_sql(sql_question)
                self._trace(f"[SQL] Generated SQL: {sql_query}")
                sql_ids = self.run_sql(sql_query)
                self._trace(f"[SQL] Matched {len(sql_ids)} row(s): {sql_ids}")
            except Exception as e:
                self._trace(f"[SQL Error] {e}")
        else:
            self._trace("[SQL] Skipped (no typed question: image-only search relies on CLIP).")

        run_vector = image_mode or include_vector_for_text
        if run_vector:
            mode = "image" if image_mode else "text (opt-in)"
            self._trace(f"[Vector] Running CLIP search, mode={mode}, threshold={VECTOR_SIMILARITY_THRESHOLD}")

            vector_hits = self.vector_search(
                text=user_input,
                image=user_image,
                k=VECTOR_SEARCH_K,
                threshold=VECTOR_SIMILARITY_THRESHOLD)

            if vector_hits:
                scored = ", ".join(f"{rid}:{score:.3f}" for rid, score in vector_hits)
                self._trace(f"[Vector] {len(vector_hits)} candidate(s) above threshold: {scored}")
            else:
                self._trace("[Vector] No candidates above the similarity threshold.")
        else:
            vector_hits = []
            self._trace("[Vector] Skipped (text-only query, 'include visual similarity' not checked).")
        vector_ids = [rid for rid, _ in vector_hits]
        vector_scores = {rid: score for rid, score in vector_hits}

        if sql_ids and not image_mode:
            sql_set = set(sql_ids)
            filtered_vector_ids = [v for v in vector_ids if v in sql_set]
            self._trace(f"[RRF] Text mode with SQL results — vector candidates restricted to SQL's rowid set: "
                        f"{filtered_vector_ids} (of {vector_ids})")
            hybrid_ids = reciprocal_rank_fusion(sql_ids, filtered_vector_ids)
        else:
            if image_mode:
                self._trace("[RRF] Image mode — SQL and vector are fused as two independent ranked lists "
                            "(rows found by both rank highest; rows found by only one still appear).")
            hybrid_ids = reciprocal_rank_fusion(sql_ids, vector_ids)

        self._trace(f"[RRF] Final fused ranking ({len(hybrid_ids)} artwork(s)): {hybrid_ids}")

        sql_set, vector_set = set(sql_ids), set(vector_ids)
        source_map = {
            rid: {"sql": rid in sql_set, "vector": rid in vector_set, "vector_score": vector_scores.get(rid)}
            for rid in hybrid_ids
        }

        context_str, image_urls, df = self.fetch_context(hybrid_ids, source_map)

        # Reset retrieval + conversation state
        self.current_context_str = context_str
        self.current_hybrid_ids = hybrid_ids
        self.current_image_urls = image_urls
        self.current_source_map = source_map
        self.last_sql_query = sql_query
        self.last_vector_ids = vector_ids
        self.last_vector_scores = vector_scores
        self.current_image_analysis = image_analysis
        self.last_search_was_image = image_mode
        self.history = []

        if not hybrid_ids:
            answer = "No artworks matched this query." + (
                f"\n\nImage analysis:\n{image_analysis}" if image_analysis else "")
            self.history = [{"role": "user", "content": user_input}, {"role": "assistant", "content": answer}]
        else:
            if user_input and user_input.strip():
                question = user_input.strip()
            elif image_mode:
                question = "Describe the retrieved artworks and how they relate to the uploaded image."
            else:
                question = "Describe and compare the retrieved artworks."
            answer = self.synthesize(question)

        self._log(mode="new_search", query=user_input, sql=sql_query, answer=answer)
        trace = self._trace_finalize()
        return {"answer": answer, "images": image_urls, "sql": sql_query, "table": df, "trace": trace,
                "vlm_description": image_analysis}


    def follow_up(self, user_question: str):
        """
        Reason over the currently retrieved context without re-running retrieval
        """

        if not self.current_context_str:
            return self.new_search(user_question)

        self._trace_reset()
        self._trace(f"[Follow-up] query={user_question!r}")
        self._trace(f"[Follow-up] No retrieval re-run. Reusing {len(self.current_hybrid_ids)} artwork(s) "
                    f"from the last search: {self.current_hybrid_ids}")
        self._trace(f"[Follow-up] Conversation history so far: {len(self.history) // 2} prior turn(s)")

        answer = self.synthesize(user_question)
        self._log(mode="follow_up", query=user_question, sql=self.last_sql_query, answer=answer)
        _, _, df = self.fetch_context(self.current_hybrid_ids, self.current_source_map)
        trace = self._trace_finalize()
        return {"answer": answer, "images": self.current_image_urls, "sql": self.last_sql_query, "table": df,
                "trace": trace, "vlm_description": self.current_image_analysis}

    def rerun_edited_sql(self, edited_sql: str):
        """
        Re-run a researcher-edited SQL string, keep prior vector ids, refresh context, resynthesize
        """

        self._trace_reset()
        self._trace(f"[Re-run SQL] Researcher-edited query: {edited_sql}")
        try:
            sql_ids = self.run_sql(edited_sql)
            self._trace(f"[Re-run SQL] Matched {len(sql_ids)} row(s): {sql_ids}")
        except Exception as e:
            self._trace(f"[Re-run SQL Error] {e}")
            return {"answer": f"SQL rejected: {e}", "images": [], "sql": edited_sql, "table": pd.DataFrame(),
                    "trace": self._trace_finalize(), "vlm_description": ""}

        sql_set = set(sql_ids)
        if sql_ids and not self.last_search_was_image:
            filtered_vector_ids = [v for v in self.last_vector_ids if v in sql_set]
            self._trace(f"[RRF] Reusing vector scores from the last search. Text mode - vector candidates"
                        f"Restricted to SQL's rowid set: {filtered_vector_ids} (of {self.last_vector_ids})")
            hybrid_ids = reciprocal_rank_fusion(sql_ids, filtered_vector_ids)
        else:
            if self.last_search_was_image:
                self._trace("[RRF] Reusing vector scores from the last (image) search. Image mode — SQL and "
                            "vector are fused as two independent ranked lists (rows found by both rank "
                            "highest; rows found by only one still appear).")
            hybrid_ids = reciprocal_rank_fusion(sql_ids, self.last_vector_ids)
        self._trace(f"[RRF] Final fused ranking ({len(hybrid_ids)} artwork(s)): {hybrid_ids}")

        vector_set = set(self.last_vector_ids)
        source_map = {
            rid: {"sql": rid in sql_set, "vector": rid in vector_set, "vector_score": self.last_vector_scores.get(rid)}
            for rid in hybrid_ids
        }

        context_str, image_urls, df = self.fetch_context(hybrid_ids, source_map)
        self.current_context_str = context_str
        self.current_hybrid_ids = hybrid_ids
        self.current_image_urls = image_urls
        self.current_source_map = source_map
        self.last_sql_query = edited_sql
        self.history = []

        if not hybrid_ids:
            answer = "No artworks matched the edited SQL query."
        else:
            answer = self.synthesize("Describe and compare the retrieved artworks.")

        self._log(mode="rerun_sql", query="(edited SQL)", sql=edited_sql, answer=answer)
        trace = self._trace_finalize()
        return {"answer": answer, "images": image_urls, "sql": edited_sql, "table": df, "trace": trace,
                "vlm_description": self.current_image_analysis}

    def _log(self, mode: str, query: str, sql: str, answer: str):
        self.session_log.append({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "query": query,
            "sql": sql,
            "answer": answer,
        })

    def export_log_csv(self) -> str:
        out_path = str(CURRENT_DIR / "session_log.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "mode", "query", "sql", "answer"])
            writer.writeheader()
            writer.writerows(self.session_log)
        return out_path


engine = ResearchEngine(conn, MAIN_TABLE, IMAGE_COLUMN, faiss_index, faiss_row_ids, embedding_model)


# 8. GRADIO FRONTEND INTERFACE


def handle_query(user_query: str, uploaded_image: Optional[Image.Image], mode: str,
                 include_vector_for_text: bool, use_caption_for_sql: bool):
    if not user_query or not user_query.strip():
        if mode == "Follow-up (reason over current results)" or uploaded_image is None:
            return (
                "Please provide a question or upload an image.",
                [],
                "",
                pd.DataFrame(),
                pd.DataFrame(engine.session_log),
                "",
                ""
            )
    try:
        image = uploaded_image.convert("RGB") if uploaded_image is not None else None
        if mode == "New search":
            result = engine.new_search(user_query, image, include_vector_for_text=include_vector_for_text,
                                       use_caption_for_sql=use_caption_for_sql)
        else:
            result = engine.follow_up(user_query)
        log_df = pd.DataFrame(engine.session_log)
        return (result["answer"], result["images"], result["sql"], result["table"], log_df,
                result.get("trace", ""), result.get("vlm_description", ""))
    except Exception as e:
        return f"Error: {e}", [], "", pd.DataFrame(), pd.DataFrame(engine.session_log), "", ""
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def handle_rerun_sql(edited_sql: str):
    try:
        result = engine.rerun_edited_sql(edited_sql)
        log_df = pd.DataFrame(engine.session_log)
        return (result["answer"], result["images"], result["sql"], result["table"], log_df,
                result.get("trace", ""), result.get("vlm_description", ""))
    except Exception as e:
        return f"Error: {e}", [], edited_sql, pd.DataFrame(), pd.DataFrame(engine.session_log), "", ""


def handle_export_log():
    return engine.export_log_csv()


with gr.Blocks(title="Rijksmuseum Research Assistant") as demo:
    gr.Markdown(
        """
        # Rijksmuseum Graphic Arts Research Assistant
        Text-to-SQL + multimodal vector search over the early modern prints and
        drawings collection. The generated SQL and retrieved records are shown below so
        you can inspect and correct retrieval, not just the final answer — including
        whether each row came from SQL, visual similarity, or both.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            mode_toggle = gr.Radio(
                ["New search", "Follow-up (reason over current results)"],
                value="New search",
                label="Mode",
            )
            user_input = gr.Textbox(label="Question", lines=3,
                                    placeholder="e.g. Which prints were produced by Cornelis Cort?")
            image_input = gr.Image(label="Upload image (optional, new search only)", type="pil")
            vector_for_text_checkbox = gr.Checkbox(
                label="Also use visual similarity search (CLIP) for this text query",
                value=False,
                info=(
                    "Off by default: CLIP text-image matching only works well for short, "
                    "topic-heavy phrasing (e.g. 'farm animals', 'landscapes'), not specific or "
                    "technical questions, so it's opt-in to avoid diluting results. Always runs "
                    "automatically when you upload an image instead."
                ),
            )
            caption_for_sql_checkbox = gr.Checkbox(
                label="Use image description to filter by metadata (experimental)",
                value=False,
                info=(
                    "Off by default. The small vision model's description is always shown to the "
                    "answer model as context, but it can misjudge medium or period, and as an SQL "
                    "filter those mistakes would silently exclude good matches. When ticked, only "
                    "two condensed keywords (object type, subject) are added to your typed question."
                ),
            )
            submit_btn = gr.Button("Submit", variant="primary")

            gr.Markdown("**Generated SQL** (editable — edit and re-run to steer retrieval directly)")
            sql_box = gr.Textbox(label="SQL", lines=4)
            rerun_btn = gr.Button("Re-run edited SQL")

            export_btn = gr.Button("Export session log (CSV)")
            export_file = gr.File(label="Session log download")

        with gr.Column(scale=2):
            output_answer = gr.Textbox(label="Answer", lines=10, interactive=False)
            output_gallery = gr.Gallery(label="Matched artwork images", columns=3, height=300)
            output_table = gr.Dataframe(label="Retrieved records", wrap=True)
            vlm_box = gr.Textbox(label="VLM description of uploaded image", lines=3, interactive=False)
            with gr.Accordion("Retrieval trace (debug)", open=False):
                trace_box = gr.Textbox(label="Step-by-step trace", lines=16, interactive=False)
            session_log_table = gr.Dataframe(label="Session log", wrap=True)

    submit_btn.click(
        fn=handle_query,
        inputs=[user_input, image_input, mode_toggle, vector_for_text_checkbox, caption_for_sql_checkbox],
        outputs=[output_answer, output_gallery, sql_box, output_table, session_log_table, trace_box, vlm_box],
    )
    rerun_btn.click(
        fn=handle_rerun_sql,
        inputs=[sql_box],
        outputs=[output_answer, output_gallery, sql_box, output_table, session_log_table, trace_box, vlm_box],
    )
    export_btn.click(fn=handle_export_log, outputs=[export_file])

if __name__ == "__main__":
    # share=True for Colab (has no accessible localhost)
    demo.queue().launch(share=False, debug=False)

