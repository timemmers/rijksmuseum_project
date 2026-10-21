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
VLM_MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"
CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

ENABLE_VLM = True # set to False on low-resource machines: image searches then run on CLIP only

CURRENT_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
DATABASE_PATH = CURRENT_DIR.parent / "preprocessing" / "rma_artworks"
# The researcher's annotations live in their own writable file; the museum database stays read-only
ANNOTATIONS_PATH = CURRENT_DIR / "annotations.db"

# Propagation: after an insight is saved, the system proposes other prints it may also apply to
PROPAGATION_SIMILARITY = 0.975 # image -> image CLIP similarity for 'same design' (0.97-0.98 separated designs in the Cort test)
PROPAGATION_SEARCH_K = 30 # nearest neighbours checked per annotated print
PROPAGATION_MAX_PER_RULE = 40 # a 'same maker and date' rule matching more prints than this is too generic to propose
PROPAGATION_MAX_TOTAL = 20 # max proposals per saved insight
# 'Visually similar' finds related compositions or motifs (insights about similarity or influence): only the closest
# few above a lower threshold. Starting values, like the other thresholds: check them against the scores in the trace
PROPAGATION_SIMILAR_MIN = 0.88
PROPAGATION_SIMILAR_MAX = 5 # max 'visually similar' proposals per saved insight
# The grounds the researcher can choose per insight. Not every insight travels the same way: a copy follows the design,
# an influence looks similar, a reissue may follow the date, an attribution may follow none of them
GROUND_DESIGN = "same design (visually near-identical)"
GROUND_SIMILAR = "visually similar (related composition or motif)"
GROUND_MAKER_DATE = "same maker and date"
PROPAGATION_GROUNDS = [GROUND_DESIGN, GROUND_SIMILAR, GROUND_MAKER_DATE]
# Ticked by default. 'Same maker and date' is not: in testing it was right for one reissue campaign and mostly noise
# for other insights (3 of 30 proposals confirmed), so the researcher switches it on when an insight follows the date
PROPAGATION_DEFAULT_GROUNDS = [GROUND_DESIGN, GROUND_SIMILAR]

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
3. The database uses Dutch terms. Translate English search terms into Dutch for LIKE clauses, including names
   (e.g. Charles -> Karel, Mary -> Maria, John -> Johannes, coronation -> kroning, flood -> zondvloed).
   Examples: 'print' -> 'prent', 'drawing' -> 'tekening', 'landscape' -> 'landschap', 'portrait' -> 'portret'.
   Subjects can only be found through the Dutch title, and titles vary, so combine several Dutch terms with OR
   inside parentheses, e.g. landscape -> ("objectTitle[1]" LIKE '%landschap%' OR "objectTitle[1]" LIKE '%zicht%').
   Each DIFFERENT element of the question must be in the title, so combine them with AND. Use OR only for
   alternative words for the SAME element, inside parentheses. E.g. "the coronation of Charles V" ->
   "objectTitle[1]" LIKE '%kroning%' AND "objectTitle[1]" LIKE '%Karel%', never kroning OR Karel.
4. Creator names are stored as 'Lastname, Firstname' (e.g. 'Cort, Cornelis'). Never match a full name as one string.
   Instead: WHERE "objectCreator[1]" LIKE '%Cornelis%' AND "objectCreator[1]" LIKE '%Cort%'
   Unattributed works have the creator 'anonymous'. Names can be stored in another form than the question uses
   (e.g. 'Gheyn, Jacob de' for Jacques de Gheyn): match on the most distinctive part of the name. When a query
   names several makers, put each maker's conditions in their own parentheses:
   (("objectCreator[1]" LIKE '%A%' AND ...) OR ("objectCreator[1]" LIKE '%B%' AND ...)).
5. If the question includes a visual description (from an uploaded image) rather than explicit search terms,
   infer plausible object type / subject / period filters from it the same way you would from a text question.
6. Researcher knowledge: the view annotated(rowid_ref, label, note, origin) holds the researcher's confirmed
   annotations; rowid_ref is the rowid of the annotated artwork. The catalogue has NO columns for publishers,
   reissues, states, copies, sources or other interpretations: these can only be found through annotated, with
   rowid IN (SELECT rowid_ref FROM annotated WHERE label LIKE '%term%' OR note LIKE '%term%')
   Never invent columns that are not in the schema.
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

Question: Show me prints of the coronation of emperor Charles V
SQL: SELECT rowid FROM {MAIN_TABLE} WHERE "objectType[1]" LIKE '%prent%' AND "objectTitle[1]" LIKE '%kroning%' AND "objectTitle[1]" LIKE '%Karel%' AND "objectCreationDate[1]" BETWEEN '1450' AND '1850';

Question: Show me 17th century portrait prints
SQL: SELECT rowid FROM {MAIN_TABLE} WHERE "objectType[1]" LIKE '%prent%' AND "objectTitle[1]" LIKE '%portret%' AND "objectCreationDate[1]" BETWEEN '1601' AND '1700';

Question: Which prints by Hendrick Goltzius are annotated as later states?
SQL: SELECT rowid FROM {MAIN_TABLE} WHERE "objectType[1]" LIKE '%prent%' AND "objectCreator[1]" LIKE '%Hendrick%' AND "objectCreator[1]" LIKE '%Goltzius%' AND rowid IN (SELECT rowid_ref FROM annotated WHERE label LIKE '%state%' OR note LIKE '%state%') AND "objectCreationDate[1]" BETWEEN '1450' AND '1850';

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
(title, object type, creator, creation date) and the researcher's annotations, never the images: do not
describe style, composition, detail, quality or technique beyond what the metadata or an annotation states.

For a NEW question, answer in this structure:
1. Answer: 1-3 sentences that answer the question directly. Take every number, date range or
   distribution about the whole result set from the exact facts about all matches, never from
   counting the records in the context.
2. Observations: 1-4 points about the records in the context (themes, series, chronology, contrasts,
   outliers), each starting with the artwork it is based on, e.g. "Artwork #4 (1559): ...".
   Only make observations the metadata or annotations support: fewer points are better than invented ones.
   Each observation must compare records, or use an annotation. Never just restate a title or call something
   "a religious scene".
3. Limits: 1-2 sentences on what these records cannot show (e.g. metadata only, or only a sample).
For a FOLLOW-UP question, answer it directly without this structure, but still cite every claim as "Artwork #N".

Guidelines:
- If the context doesn't contain enough information to answer confidently, say so explicitly
  rather than guessing.
- If the records in the context do not match what the question asks for (e.g. the search found other subjects
  or other makers), say so plainly and suggest refining the search. Never reinterpret an unrelated title to
  make it fit the question.
- Never claim that something is not in the database: you only see the results of this search. Say that this
  search did not find it, and that the search terms or name forms may need adjusting.
- If a question concerns records that are not in the context (e.g. a subset of all matches),
  say that a refined or new search is needed instead of answering from the context.
- A date in a title usually refers to the depicted event or place, not to when the object was made
  (e.g. a battle of 1693 in a print made in 1702); use the creation date for the object itself.
- The context may end with a RESEARCHER ANNOTATIONS block. These are knowledge the researcher wrote or
  confirmed, not catalogue metadata, and the most important context for interpretation. When annotations
  are relevant to the question, use them already in the Answer, not only in the Observations. Cite them as
  "Annotation A<n>" next to the artwork (e.g. "Artwork #2 (Annotation A1): ..."), and say so if the records
  seem to contradict them.
- Write for a researcher: precise, willing to flag uncertainty, comfortable making a reasoned
  comparative judgment rather than only describing.

Database Context (retrieved artworks for this session):
{context}
"""

# Drafting only restructures what the researcher said; the researcher checks and saves the draft
ANNOTATION_DRAFT_SYS = """You turn a researcher's insight about artworks in a print collection into a structured
annotation. You get the artworks currently on screen and the researcher's own words.

Reply with ONLY a JSON object, nothing else:
{"artworks": [<artwork numbers>], "label": "<short label>", "note": "<note>"}

RULES:
1. artworks: the numbers of the artworks the insight is about, chosen ONLY from the list on screen.
   Resolve references such as '#2', 'the 1602 print' or 'the earlier impressions' using the list.
2. label: 2-5 words naming the insight, e.g. "later state".
3. note: the researcher's insight in one or two clear sentences. Use ONLY what the researcher said:
   never add facts, dates, names, counts or interpretations they did not mention, and never make the note
   more specific than their words.
4. Never refer to artworks by their "Artwork #" number in the note: these numbers change with every search.
   If the note must refer to another artwork, use its inventory number.
5. If it is unclear which artworks are meant, use an empty list. Select all artworks only if the
   researcher explicitly says the insight applies to all of them.

EXAMPLE:
Artworks on screen:
Artwork #1 | RP-P-OB-1001 | Landschap met jagers | 1575
Artwork #2 | RP-P-OB-1002 | Landschap met jagers | 1610
Researcher: #2 is a later state of #1, the address was changed
JSON: {"artworks": [2], "label": "later state", "note": "A later state of the plate of RP-P-OB-1001, with a changed address."}
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


# 6b. RESEARCHER ANNOTATIONS
# The researcher's insights, stored in their own writable SQLite file (ANNOTATIONS_PATH), never in the
# museum database. One annotation can cover several artworks ("these are reissues of one plate"), so there
# are two tables: the insight itself, and which records it applies to. 'origin' and 'status' record the
# provenance: researcher insights are saved as confirmed; system proposals (origin 'system') stay 'proposed'
# until the researcher confirms or rejects them

def open_annotations(path: Path):
    """
    Opens (or creates) the annotations database
    """

    # check_same_thread=False: Gradio handles requests in different threads
    ann_conn = sqlite3.connect(str(path), check_same_thread=False)
    ann_conn.execute("PRAGMA foreign_keys = ON")
    ann_conn.executescript("""
        CREATE TABLE IF NOT EXISTS annotations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            label       TEXT,
            note        TEXT NOT NULL,
            origin      TEXT NOT NULL CHECK (origin IN ('researcher', 'system')),
            status      TEXT NOT NULL CHECK (status IN ('proposed', 'confirmed', 'rejected')),
            based_on    INTEGER REFERENCES annotations(id),
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS annotation_records (
            annotation_id  INTEGER NOT NULL REFERENCES annotations(id),
            rowid_ref      INTEGER NOT NULL,
            PRIMARY KEY (annotation_id, rowid_ref)
        );
    """)
    # 'rowid_ref', not 'rowid': every SQLite table already has a hidden rowid column, which a column
    # with the same name would shadow. rowid_ref REFERS to a rowid in the museum database

    # 'reason': why the SYSTEM proposed an annotation. Added this way (not in CREATE TABLE) so that annotation
    # files created before this column existed are upgraded instead of failing
    columns = [col[1] for col in ann_conn.execute("PRAGMA table_info(annotations)")]
    if "reason" not in columns:
        with ann_conn:
            ann_conn.execute("ALTER TABLE annotations ADD COLUMN reason TEXT")
    print(f"[Annotations] Using '{Path(path).resolve()}'") # the full path, so you can see where the file really is
    return ann_conn


def add_annotation(ann_conn, rowids: list, note: str, label: Optional[str] = None,
                   origin: str = "researcher", status: str = "confirmed", based_on: Optional[int] = None,
                   reason: Optional[str] = None) -> int:
    """
    Saves one insight about one or more records; returns its id
    """

    now = datetime.datetime.now().isoformat(timespec="seconds")
    with ann_conn: # one transaction: either the annotation AND its records are saved, or nothing
        cursor = ann_conn.execute(
            "INSERT INTO annotations (label, note, origin, status, based_on, reason, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (label, note, origin, status, based_on, reason, now, now),
        )
        annotation_id = cursor.lastrowid
        ann_conn.executemany("INSERT INTO annotation_records (annotation_id, rowid_ref) VALUES (?, ?)",
                             [(annotation_id, int(r)) for r in rowids])
    return annotation_id


def all_annotations(ann_conn) -> dict:
    """
    All CONFIRMED annotations, grouped per museum rowid: proposals only count as knowledge once the researcher
    confirms them. The table stays small, so loading all of it is simpler and safer than an IN (...) query over
    thousands of matches
    """

    rows = ann_conn.execute("""
        SELECT ar.rowid_ref, a.id, a.label, a.note, a.origin, a.status, a.reason
        FROM annotation_records ar
        JOIN annotations a ON a.id = ar.annotation_id
        WHERE a.status = 'confirmed'
        ORDER BY a.created_at
    """).fetchall()

    by_rowid = {}
    for rowid_ref, ann_id, label, note, origin, status, reason in rows:
        by_rowid.setdefault(int(rowid_ref), []).append(
            {"id": ann_id, "label": label, "note": note, "origin": origin, "status": status, "reason": reason})
    return by_rowid


def set_annotation_status(ann_conn, annotation_ids: list, status: str):
    """
    Confirms or rejects annotations (e.g. system proposals). Rejected ones are kept, not deleted, so the
    history stays visible and they are never proposed again
    """

    now = datetime.datetime.now().isoformat(timespec="seconds")
    with ann_conn:
        ann_conn.executemany("UPDATE annotations SET status = ?, updated_at = ? WHERE id = ?",
                             [(status, now, int(i)) for i in annotation_ids])


def attach_annotations(db_conn, path: Path):
    """
    Makes CONFIRMED annotations searchable by the generated SQL: the annotations file is attached READ-ONLY to
    the museum connection, with a view 'annotated'. Writing still only happens through the annotations
    connection, so no generated query can change an annotation
    """

    uri = f"{Path(path).resolve().as_uri()}?mode=ro" # the museum connection was opened with uri=True
    db_conn.execute("ATTACH DATABASE ? AS ann", (uri,))
    # TEMP: the view lives in the connection's temporary schema, so the read-only museum file is untouched
    db_conn.execute("""
        CREATE TEMP VIEW annotated AS
        SELECT ar.rowid_ref AS rowid_ref, a.label AS label, a.note AS note, a.origin AS origin
        FROM ann.annotation_records ar
        JOIN ann.annotations a ON a.id = ar.annotation_id
        WHERE a.status = 'confirmed'
    """)
    print("[Annotations] Confirmed annotations are searchable by SQL (view 'annotated', read-only)")


def _annotation_text(ann: dict) -> str:
    return f"{ann['label']}: {ann['note']}" if ann["label"] else ann["note"]


# 7. PROTOTYPE ENGINE

DANGEROUS_KEYWORDS = ["drop", "delete", "update", "insert", "alter", "create", "truncate", "attach", "pragma"]
DANGEROUS_PATTERN = re.compile(r"\b(" + "|".join(DANGEROUS_KEYWORDS) + r")\b", re.IGNORECASE)


def _preview_ids(ids: list, n: int = 50) -> str:
    """
    Shortens long rowid lists for the trace
    """
    return str(ids) if len(ids) <= n else f"{ids[:n]} ... (+{len(ids) - n} more)"


def _is_filled(value) -> bool:
    return pd.notna(value) and str(value).strip() not in ("", "<null>")


def _year(value) -> float:
    """
    The first four-digit year in a date, so '1602', '1525-04-11' and 'ca. 1570' all work.
    pd.to_numeric would turn full dates like '1525-04-11' into 'no value', which sorted them last
    """

    match = re.search(r"(?<!\d)(\d{4})(?!\d)", str(value)) if _is_filled(value) else None
    return float(match.group(1)) if match else float("nan")


class ResearchEngine:
    """
    Holds retrieval state (current results) and conversation history separately,
    so a follow-up question can reason over the same retrieved artworks without
    re-running SQL/vector search, and a "new search" explicitly resets both

    Results are kept at two levels:
    - self.results: ALL records that matched (the 'All matches' table, the profile and the CSV export)
    - self.context_df: the sample of those records the answer LLM reads in detail ("Artwork #1" ...)

    The researcher's annotations (separate database) are attached to both, and to the LLM's context
    """

    def __init__(self, db_conn, table_name, faiss_idx, row_ids_arr, embed_model, ann_conn):
        self.db_conn = db_conn
        self.table_name = table_name
        # Column names of the museum table, so column names taken from generated SQL can be checked
        self.columns = [col[1] for col in db_conn.execute(f"PRAGMA table_info('{table_name}')")]
        self.faiss_index = faiss_idx
        self.row_ids = row_ids_arr
        self.embed_model = embed_model
        self.ann_conn = ann_conn

        # rowid -> position in the FAISS index, so any set of rows (e.g. SQL hits) can be scored directly
        self.rowid_to_pos = {int(r): i for i, r in enumerate(row_ids_arr)}

        # Retrieval state (persists across follow-up turns)
        self.results = pd.DataFrame()
        self.context_df = pd.DataFrame()
        self.context_str = ""
        self.images = []
        self.profile = ""
        self.ranking_note = "" # how the sample was chosen; told to the answer LLM

        # Annotations: per rowid (all of them), and the ones in the current context (id -> annotation),
        # so citations like "Annotation A1" can be checked
        self.annotations_by_rowid = {}
        self.context_annotations = {}
        self.last_cited_annotations = [] # annotations the last answer mentioned (for the log)

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

    def retry_sql(self, question: str, failed_sql: str, counts: list) -> str:
        """
        One retry after a query found nothing, with the term counts as feedback: the DATA shows which search terms
        don't occur in the catalogue, instead of the model guessing why nothing was found
        """

        feedback = "\n".join(f"- {col} contains '{pattern.strip('%')}': {n} record(s)" for col, pattern, n in counts)
        messages = [
            {"role": "system", "content": SQL_FILTER_SYS},
            {"role": "user", "content": (
                f"Question: {question}\n"
                f"This query found nothing:\n{failed_sql};\n"
                f"Number of records in the whole catalogue that contain each search term:\n{feedback}\n"
                "Terms with 0 records do not occur in the catalogue. Write a new query that replaces them with other "
                "Dutch words for the same element (a synonym, another spelling, the Dutch form of a name, or a shorter "
                "word stem), and keeps the other conditions.\nSQL query:")},
        ]
        return self._clean_sql(generate_chat(messages, max_new_tokens=250, stop_strings=[";"]))

    def term_counts(self, sql_text: str, limit: int = 8) -> list:
        """
        For every "column" LIKE '...' condition in a query: how many records in the whole catalogue match it on
        its own. Computed by the database, so it shows which words of a query actually occur in the catalogue
        Returns (column, pattern, count) triples
        """

        conditions = []
        for col, pattern in re.findall(r"\"([^\"]+)\"\s+LIKE\s+'([^']*)'", sql_text, flags=re.IGNORECASE):
            if col in self.columns and (col, pattern) not in conditions: # only real columns, never raw SQL
                conditions.append((col, pattern))
        return [(col, pattern, int(self.db_conn.execute(
                    f'SELECT COUNT(*) FROM {self.table_name} WHERE "{col}" LIKE ?', (pattern,)).fetchone()[0]))
                for col, pattern in conditions[:limit]]

    @staticmethod
    def _term_note(counts: list) -> str:
        return ("records in the catalogue containing each search term: " +
                "; ".join(f"'{pattern.strip('%')}' ({col}): {n}" for col, pattern, n in counts)) if counts else ""

    def _check_empty_result(self, sql_text: str, question: Optional[str] = None):
        """
        SQL guardrail for a query that found nothing. The term counts show which words don't occur in the
        catalogue. With a question (a generated query), the model gets one retry with the counts as feedback;
        without one (the researcher's own SQL), only the counts are shown and the query is never rewritten
        Returns (sql, sql_ids, sql_for_ids, note for the researcher)
        """

        counts = self.term_counts(sql_text)
        note = self._term_note(counts)
        self._trace(f"[Terms] Nothing found. {note or 'No LIKE conditions to check.'}")
        missing = [pattern.strip('%') for _, pattern, n in counts if n == 0]
        if not missing:
            hint = (" Every term occurs on its own, so the combination is too narrow." if counts else "")
            return sql_text, [], "", f"NOTE: the query found nothing.{hint} {note}".strip()
        if question is None:
            return sql_text, [], "", (f"NOTE: the query found nothing; these terms don't occur in the catalogue: "
                                      f"{', '.join(missing)}. {note}")

        retried = self.retry_sql(question, sql_text, counts)
        self._trace(f"[Terms] Retry with the term counts as feedback: {retried}")
        try:
            sql_ids, sql_for_ids = self.run_sql(retried)
        except Exception as e:
            self._trace(f"[SQL Error] Retry rejected: {e}")
            return sql_text, [], "", f"NOTE: the query found nothing. {note}"
        self._trace(f"[Terms] Retry matched {len(sql_ids)} row(s): {_preview_ids(sql_ids)}")
        if sql_ids:
            return retried, sql_ids, sql_for_ids, (
                f"NOTE: the first query found nothing, because these terms don't occur in the catalogue: "
                f"{', '.join(missing)}. The system retried once with other terms: check the SQL shown.")
        return retried, [], "", (f"NOTE: the query found nothing, also after one retry. First query: {note}. "
                                 f"Retry: {self._term_note(self.term_counts(retried))}")

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
        df["year"] = df[DATE_COLUMN].map(_year) # numeric year for sorting and the profile
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
            self.annotations_by_rowid, self.context_annotations = {}, {}
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

        # The researcher's annotations for these records, then the LLM's context and the gallery
        self._refresh_annotations()
        self._build_context()

    def _refresh_annotations(self):
        """
        Reads the researcher's annotations and shows them in an 'annotations' column of both tables
        """

        self.annotations_by_rowid = all_annotations(self.ann_conn)
        n_annotated = 0
        for df in (self.results, self.context_df):
            if df.empty:
                continue
            values = df["rowid"].map(lambda r: "; ".join(
                f"A{a['id']}: {a['label'] or a['note'][:40]}" for a in self.annotations_by_rowid.get(int(r), [])))
            if "annotations" in df.columns:
                df["annotations"] = values
            else:
                df.insert(3, "annotations", values)
            n_annotated = int((values != "").sum()) # the last loop is the context
        self._trace(f"[Annotations] {n_annotated} of the {len(self.context_df)} artwork(s) in the context "
                    f"carry a researcher annotation.")

    def _build_context(self):
        """
        Context for the LLM (object number as identifier, plus the researcher's annotations) and captioned
        images for the gallery. Separate from retrieve(), so a saved annotation can update the context
        without searching again
        """

        context_parts, self.images, self.context_annotations = [], [], {}
        for _, row in self.context_df.iterrows():
            n, obj = row["Artwork #"], row[OBJECT_NUMBER_COLUMN]
            details = " | ".join(f"{label}: {row[col]}" for col, label in CONTEXT_FIELDS if _is_filled(row[col]))
            anns = self.annotations_by_rowid.get(int(row["rowid"]), [])
            for ann in anns:
                # Remember on which artworks in the context each annotation appears (for the block and the sources)
                self.context_annotations.setdefault(ann["id"], {**ann, "artworks": []})["artworks"].append(n)
            # The label inline, so the annotation is visible where the record is (the full note is in the block)
            ann_labels = ", ".join(f"A{a['id']} ({a['label'] or a['note'][:40]})" for a in anns)
            marker = f" | annotated: {ann_labels}" if anns else ""
            context_parts.append(f"[Artwork #{n} | {obj}] {details}{marker}")
            if str(row[IMAGE_COLUMN]).startswith("http"):
                self.images.append((row[IMAGE_COLUMN], f"Artwork #{n} ({obj}) - {row[TITLE_COLUMN]}"))

        # Each annotation once, in its own block: small models pick up a labelled section more reliably
        # than lines repeated under several artworks
        if self.context_annotations:
            block = ["RESEARCHER ANNOTATIONS (the researcher's own confirmed knowledge, not catalogue data):"]
            for ann_id, ann in self.context_annotations.items():
                on = ", ".join(f"#{n}" for n in ann["artworks"])
                why = f"; proposed by the system because: {ann['reason']}" if ann["origin"] == "system" else ""
                block.append(f"A{ann_id} on Artwork {on} ({ann['origin']}, {ann['status']}{why}): "
                             f"{_annotation_text(ann)}")
            context_parts.append("\n".join(block))
        self.context_str = "\n\n".join(context_parts)

    # Annotations

    def save_annotation(self, selected: list, label: str, note: str, grounds: Optional[list] = None) -> str:
        """
        Saves a researcher insight about one or more artworks in the context. It is stored in the annotations
        database (never in the museum database), shown in the tables, and from now on given to the answer LLM
        with every search or follow-up that retrieves these records. 'grounds' are the propagation grounds the
        researcher chose for this insight (None = the default grounds)
        """

        numbers = sorted({int(m) for s in (selected or []) for m in re.findall(r"Artwork #(\d+)", s)})
        if not numbers:
            return "Select at least one artwork first."
        if not (note or "").strip():
            return "Write a note first."

        rows = self.context_df[self.context_df["Artwork #"].isin(numbers)]
        label = (label or "").strip() or None
        ann_id = add_annotation(self.ann_conn, [int(r) for r in rows["rowid"]], note.strip(), label=label)

        # Update the tables and the LLM's context without searching again, so a follow-up already uses it
        self._refresh_annotations()
        self._build_context()

        artworks = ", ".join(f"#{n}" for n in numbers)
        objects = ", ".join(rows[OBJECT_NUMBER_COLUMN].astype(str))
        self._trace(f"[Annotations] Saved A{ann_id} on Artwork {artworks} ({objects})")
        self._log("annotate", f"A{ann_id} on Artwork {artworks} ({objects})", "",
                  _annotation_text({"label": label, "note": note.strip()}))

        # The system's contribution: other prints this insight may also apply to (proposals, not knowledge)
        grounds = PROPAGATION_DEFAULT_GROUNDS if grounds is None else list(grounds)
        if not grounds:
            self._trace(f"[Propagation] A{ann_id}: no grounds chosen, so no proposals.")
            return (f"Saved as A{ann_id} on Artwork {artworks}. It is now part of the model's context. "
                    f"No proposals: no grounds were chosen.")
        n_proposed = self.propose_extensions(ann_id, rows, label, note.strip(), grounds)
        proposed = (f" The system proposes {n_proposed} other print(s) it may also apply to: check them under "
                    f"'Proposed extensions'." if n_proposed else " The system found no other prints to propose.")
        return f"Saved as A{ann_id} on Artwork {artworks}. It is now part of the model's context.{proposed}"

    def propose_extensions(self, ann_id: int, rows: pd.DataFrame, label: Optional[str], note: str,
                           grounds: Optional[list] = None) -> int:
        """
        Proposes other prints a saved insight may also apply to, on three grounds (each proposal records which):
        - same design: visually near-identical prints in the CLIP index (other impressions, states or copies).
          Note that impressions from one plate look identical whatever their date
        - visually similar: the few closest prints above a lower threshold (related compositions or motifs), for
          insights about similarity or influence rather than identity
        - same maker and date: other prints with exactly the same creator and creation date (e.g. one reissue
          campaign), which visual similarity cannot separate from the originals
        Only the grounds the researcher chose for this insight are used (None = the default grounds).
        Proposals are saved as origin 'system', status 'proposed', and only count once the researcher confirms them
        """

        grounds = PROPAGATION_DEFAULT_GROUNDS if grounds is None else list(grounds)

        # Prints that already carry this insight, or were already proposed for it (incl. rejected ones)
        already = {int(r) for (r,) in self.ann_conn.execute(
            "SELECT ar.rowid_ref FROM annotation_records ar JOIN annotations a ON a.id = ar.annotation_id "
            "WHERE a.id = ? OR a.based_on = ?", (ann_id, ann_id))}
        candidates = {} # rowid -> reason (the first reason found wins)

        # Visual neighbours of the annotated prints, once for both visual grounds: (candidate, score, source print)
        neighbours = self._visual_neighbours(rows, already) if (
            GROUND_DESIGN in grounds or GROUND_SIMILAR in grounds) else []

        # 1. Same design: near-identical neighbours
        if GROUND_DESIGN in grounds:
            for cand, score, source in neighbours:
                if score >= PROPAGATION_SIMILARITY:
                    candidates.setdefault(cand, f"same design: visually near-identical to {source} "
                                                f"(similarity {score:.3f})")

        # 2. Visually similar: the closest few above the lower threshold, not already proposed as 'same design'
        if GROUND_SIMILAR in grounds:
            similar = [(cand, score, source) for cand, score, source in neighbours
                       if score >= PROPAGATION_SIMILAR_MIN and cand not in candidates]
            for cand, score, source in similar[:PROPAGATION_SIMILAR_MAX]:
                candidates.setdefault(cand, f"visually similar to {source} (similarity {score:.3f})")

        # 3. Same maker and date: exact matches in the metadata
        maker_dates = set(zip(rows[CREATOR_COLUMN], rows[DATE_COLUMN])) if GROUND_MAKER_DATE in grounds else set()
        for creator, date in maker_dates:
            if not (_is_filled(creator) and _is_filled(date)) or str(creator).lower() in ("anoniem", "anonymous"):
                continue # without a named maker and a date this rule would match far too much
            matches = [int(r) for (r,) in self.db_conn.execute(
                f'SELECT rowid FROM {self.table_name} WHERE "{CREATOR_COLUMN}" = ? AND "{DATE_COLUMN}" = ?',
                (creator, date)) if int(r) not in already]
            if len(matches) > PROPAGATION_MAX_PER_RULE:
                self._trace(f"[Propagation] Skipped 'same maker and date' ({creator}, {date}): {len(matches)} "
                            f"prints is too generic to propose.")
                continue
            for cand in matches:
                candidates.setdefault(cand, f"same maker and date: {creator}, {date}")

        selected = list(candidates.items())[:PROPAGATION_MAX_TOTAL]
        for cand, reason in selected:
            add_annotation(self.ann_conn, [cand], note, label=label, origin="system", status="proposed",
                           based_on=ann_id, reason=reason)
        self._trace(f"[Propagation] A{ann_id}: {len(selected)} proposal(s) saved"
                    + (f" ({len(candidates) - len(selected)} more left out, max {PROPAGATION_MAX_TOTAL})"
                       if len(candidates) > len(selected) else ""))
        self._log("propose", f"A{ann_id} (grounds: {', '.join(grounds)})", "",
                  "; ".join(f"{c}: {r}" for c, r in selected))
        return len(selected)

    def _visual_neighbours(self, rows: pd.DataFrame, exclude: set) -> list:
        """
        Nearest neighbours in the visual index of each annotated print, as (rowid, similarity, inventory number of
        the annotated print it resembles), best score per candidate, sorted from most to least similar
        """

        if not self.faiss_index.ntotal:
            return []
        best = {} # candidate -> (score, source)
        for _, row in rows.iterrows():
            pos = self.rowid_to_pos.get(int(row["rowid"]))
            if pos is None:
                continue # this print has no image in the index
            vec = np.ascontiguousarray(self.faiss_index.reconstruct(pos), dtype="float32").reshape(1, -1)
            scores, indices = self.faiss_index.search(vec, PROPAGATION_SEARCH_K)
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                cand = int(self.row_ids[idx])
                if cand not in exclude and score > best.get(cand, (-1.0, ""))[0]:
                    best[cand] = (float(score), row[OBJECT_NUMBER_COLUMN])
        return sorted(((c, sc, src) for c, (sc, src) in best.items()), key=lambda x: x[1], reverse=True)

    def pending_proposals(self, limit: int = 60) -> list:
        """
        The system's open proposals, with the metadata the researcher needs to judge them
        """

        rows = self.ann_conn.execute("""
            SELECT a.id, ar.rowid_ref, a.label, a.reason, a.based_on
            FROM annotations a
            JOIN annotation_records ar ON ar.annotation_id = a.id
            WHERE a.origin = 'system' AND a.status = 'proposed'
            ORDER BY a.id
            LIMIT ?
        """, (limit,)).fetchall()
        if not rows:
            return []

        rowids = [int(r[1]) for r in rows]
        placeholders = ",".join("?" for _ in rowids)
        meta = {int(r[0]): r[1:] for r in self.db_conn.execute(
            f'SELECT rowid, "{OBJECT_NUMBER_COLUMN}", "{TITLE_COLUMN}", "{DATE_COLUMN}", "{IMAGE_COLUMN}" '
            f'FROM {self.table_name} WHERE rowid IN ({placeholders})', rowids)}
        return [{"id": ann_id, "rowid": int(rid), "label": label, "reason": reason, "based_on": based_on,
                 "object": meta[int(rid)][0], "title": meta[int(rid)][1], "date": meta[int(rid)][2],
                 "image": meta[int(rid)][3]}
                for ann_id, rid, label, reason, based_on in rows if int(rid) in meta]

    def resolve_proposals(self, selected: list, status: str) -> str:
        """
        The researcher confirms (the insight now applies to these prints) or rejects (it doesn't) proposals
        """

        ids = sorted({int(m) for s in (selected or []) for m in re.findall(r"^A(\d+)", s)})
        if not ids:
            return "Select at least one proposal first."
        set_annotation_status(self.ann_conn, ids, status)

        # Confirmed proposals are knowledge now: update the tables and the LLM's context
        self._refresh_annotations()
        self._build_context()
        listed = ", ".join(f"A{i}" for i in ids)
        self._trace(f"[Propagation] {status.capitalize()}: {listed}")
        self._log("confirm_proposals" if status == "confirmed" else "reject_proposals", listed, "", "")
        return f"{status.capitalize()} {len(ids)} proposal(s): {listed}."

    def draft_annotation(self, researcher_text: str) -> dict:
        """
        The LLM turns the researcher's own words into a draft annotation (artworks, label, note). Nothing is
        saved here: the draft fills the panel, and the researcher checks, edits and saves it (human authority)
        """

        empty = {"numbers": [], "label": "", "note": ""}
        if self.context_df.empty:
            return {**empty, "message": "Search first: there are no artworks in the context."}
        if not (researcher_text or "").strip():
            return {**empty, "message": "Describe your insight first."}

        artworks = "\n".join(f"Artwork #{row['Artwork #']} | {row[OBJECT_NUMBER_COLUMN]} | {row[TITLE_COLUMN]} | "
                             f"{row[DATE_COLUMN]}" for _, row in self.context_df.iterrows())
        messages = [
            {"role": "system", "content": ANNOTATION_DRAFT_SYS},
            {"role": "user", "content": f"Artworks on screen:\n{artworks}\nResearcher: {researcher_text.strip()}\nJSON:"},
        ]
        raw = generate_chat(messages, max_new_tokens=200) # deterministic: a draft should not be creative
        self._trace(f"[Annotations] Draft for {researcher_text!r}: {raw}")

        # Small models sometimes wrap the JSON in text or ``` fences, so take the first {...} block
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        try:
            draft = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            draft = {}
        if not isinstance(draft, dict):
            draft = {}

        # Keep only artwork numbers that really are in the context (also accepts "#2" or "2")
        valid = {int(n) for n in self.context_df["Artwork #"]}
        numbers = sorted({int(d) for n in draft.get("artworks", []) or [] for d in re.findall(r"\d+", str(n))
                          if int(d) in valid})

        # The draft is logged too, so the session log shows what the model proposed and what was saved
        self._log("draft_annotation", researcher_text.strip(), "", raw)

        if not draft:
            message = "The model's draft could not be read: fill in the fields yourself."
        elif not numbers:
            message = "Draft ready, but it's unclear which artworks are meant: tick them yourself, then Save."
        else:
            message = "Draft ready: check the artworks, label and note, edit if needed, then Save. Nothing is saved yet."
        return {"numbers": numbers, "label": str(draft.get("label", "") or ""),
                "note": str(draft.get("note", "") or ""), "message": message}

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
        if self.context_annotations:
            ids = ", ".join(f"A{a}" for a in self.context_annotations)
            system_content += (f"\nThe context contains {len(self.context_annotations)} annotation(s) written or "
                               f"confirmed by the researcher: {ids}. Use them in your answer and cite them.\n")
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
        records from the database and the cited researcher annotations (so every citation can be checked),
        and warnings for invalid or missing citations
        """

        n_read, n_total = len(self.context_df), len(self.results)
        scope = (f"[Based on all {n_total} matching records.]" if n_read >= n_total else
                 f"[Based on {n_read} of {n_total} matching records: {self.ranking_note}. "
                 f"Exact figures over all matches are in the profile.]")

        # Case-insensitive: small models also write "artwork #4"
        cited = sorted({int(n) for n in re.findall(r"Artwork #(\d+)", answer, flags=re.IGNORECASE)})
        lines = []
        for n in cited:
            match = self.context_df[self.context_df["Artwork #"] == n]
            if match.empty:
                lines.append(f"WARNING: the answer cites Artwork #{n}, which is not in the context.")
                continue
            row = match.iloc[0]
            lines.append(f"Artwork #{n}: " + " | ".join(str(row[c]) for c in (
                OBJECT_NUMBER_COLUMN, TITLE_COLUMN, CREATOR_COLUMN, DATE_COLUMN, HANDLE_COLUMN) if _is_filled(row[c])))

        # Cited annotations: shown with their provenance, so researcher knowledge is never mistaken for catalogue data
        # Also bare "A6": the model doesn't always write "Annotation A6"
        mentioned = {int(a) for a in re.findall(r"(?<![-\w])(?:Annotation\s+)?A(\d+)\b", answer)}
        for a in sorted(mentioned):
            ann = self.context_annotations.get(a)
            if ann is None:
                lines.append(f"WARNING: the answer cites Annotation A{a}, which is not in the context.")
                continue
            on = ", ".join(f"#{n}" for n in ann["artworks"])
            lines.append(f"Annotation A{a} ({ann['origin']}, {ann['status']}, on Artwork {on}): {_annotation_text(ann)}")

        if not cited:
            lines.append("WARNING: the answer cites no specific records; its claims are not grounded in the data.")
        self._trace("[Check] " + ("; ".join(l for l in lines if l.startswith("WARNING")) or
                                  "All cited artwork numbers exist in the context."))

        # Which annotations in the context the answer actually mentioned (logged, and marked in the block below)
        self.last_cited_annotations = sorted(mentioned & set(self.context_annotations))
        if self.context_annotations:
            self._trace(f"[Check] Annotations in the context: {len(self.context_annotations)}; mentioned in the "
                        f"answer: {len(self.last_cited_annotations)}")

        return (scope + "\n\n" + answer + "\n\n---\nSources (from the database, not generated by the model):\n"
                + "\n".join(lines) + self._known_block(set(self.last_cited_annotations)))

    def _known_block(self, mentioned: set) -> str:
        """
        Every annotation on the records in the context, grouped by label and written by the CODE, not the model:
        the researcher always sees the relevant knowledge, also when the answer leaves it out. Groups the answer
        didn't mention are marked, so a gap between what is known and what was said is visible
        """

        if not self.context_annotations:
            return ""
        groups = {}
        for ann_id, ann in self.context_annotations.items():
            key = (ann["label"] or ann["note"][:40]).strip().lower()
            groups.setdefault(key, []).append((ann_id, ann))

        lines = ["", "", "Known about these records (annotations, listed by the code, not generated by the model):"]
        for anns in groups.values():
            first = anns[0][1]
            label = first["label"] or first["note"][:40]
            artworks = ", ".join(f"#{n}" for n in sorted({n for _, a in anns for n in a["artworks"]}))
            ids = ", ".join(f"A{i}" for i, _ in anns)
            n_researcher = sum(1 for _, a in anns if a["origin"] == "researcher")
            n_system = len(anns) - n_researcher
            origin = "; ".join(part for part in (
                f"{n_researcher} by the researcher" if n_researcher else "",
                f"{n_system} proposed by the system and confirmed" if n_system else "") if part)
            # The researcher's own note describes the insight best, if there is one
            note = next((a["note"] for _, a in anns if a["origin"] == "researcher"), first["note"])
            flag = "" if any(i in mentioned for i, _ in anns) else " [not mentioned in the answer]"
            lines.append(f"- {label}: Artwork {artworks} ({ids}; {origin}){flag}")
            lines.append(f"  {note}")
        return "\n".join(lines)

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

        sql, sql_ids, sql_for_ids, term_note = "", [], "", ""
        if sql_question:
            try:
                sql = self.generate_sql(sql_question)
                self._trace(f"[SQL] Generated SQL: {sql}")
                sql_ids, sql_for_ids = self.run_sql(sql)
                self._trace(f"[SQL] Matched {len(sql_ids)} row(s): {_preview_ids(sql_ids)}")
                if not sql_ids: # SQL guardrail: which terms don't occur in the catalogue, then one retry
                    sql, sql_ids, sql_for_ids, term_note = self._check_empty_result(sql, sql_question)
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
        return self._finish("new_search", user_input, self.last_question, sql, term_note)

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
        if not sql_ids:
            warning = self._check_empty_result(refined_sql)[3] # term counts only, no retry
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

        # The researcher's own query is never rewritten; an empty result only gets the term counts
        warning = self._check_empty_result(edited_sql)[3] if not sql_ids else ""
        self.retrieve(sql_ids, sql_for_ids, self.last_vector_hits, self.last_search_was_image, self.last_query_vec)
        self.last_sql, self.last_sql_ids = edited_sql, sql_ids
        return self._finish("rerun_sql", "(edited SQL)", "Describe and compare the retrieved artworks.", edited_sql,
                            warning)

    def _finish(self, mode: str, log_query: str, question: str, sql: str, warning: str = ""):
        """
        Common end of every retrieval: reset the conversation, answer (or report no matches), log
        """

        self.history = []
        self.last_cited_annotations = [] # stays empty when there is no answer, so the log doesn't repeat old citations
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
            # Which annotations the model had in its context, so the log shows whether it COULD have used them
            "context_annotations": ";".join(f"A{a}" for a in self.context_annotations),
            # ...and which of them the answer actually mentioned (context vs mentioned = annotation use)
            "cited_annotations": ";".join(f"A{a}" for a in self.last_cited_annotations)
                                 if mode in ("new_search", "refine", "follow_up", "rerun_sql") else "",
            "llm": LLM_MODEL_NAME,
        })

    def export_log_csv(self) -> str:
        out_path = str(CURRENT_DIR / "session_log.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "mode", "query", "sql", "answer", "total_matches",
                                                   "ranking_note", "context_records", "context_annotations",
                                                   "cited_annotations", "llm"])
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


annotations_conn = open_annotations(ANNOTATIONS_PATH) # creates the file if needed, so it can be attached
attach_annotations(conn, ANNOTATIONS_PATH)
engine = ResearchEngine(conn, MAIN_TABLE, faiss_index, faiss_row_ids, embedding_model, annotations_conn)


# 8. GRADIO FRONTEND INTERFACE


def _artwork_choices() -> list:
    # One checkbox per artwork in the model's context, e.g. 'Artwork #2 - Presentatie van Maria'
    if engine.context_df.empty:
        return []
    return [f"Artwork #{n} - {str(t)[:60]}" for n, t in
            zip(engine.context_df["Artwork #"], engine.context_df[TITLE_COLUMN])]


def _outputs(result: dict) -> tuple:
    # The two table titles state the actual numbers of each search, so the boxes can't be misread
    n_all, n_context = result["n_matches"], len(result["table"])
    context_table = gr.Dataframe(value=result["table"],
                                 label=f"Records in the model's context: {n_context} of {n_all} matches (what the AI reads)")
    all_table = gr.Dataframe(value=result["all_matches"], label=f"All matches: {n_all} records found by the search" + (
        f" (first {MAX_RESULT_ROWS_DISPLAY} shown here; the CSV export has all)" if n_all > MAX_RESULT_ROWS_DISPLAY else ""))
    # The annotation checkboxes always list the artworks currently in the context
    artwork_boxes = gr.CheckboxGroup(choices=_artwork_choices() if result["n_matches"] else [], value=[])
    return (result["answer"], result["images"], result["sql"], context_table, pd.DataFrame(engine.session_log),
            result["trace"], result["vlm_description"], result["profile"], all_table, artwork_boxes)


def handle_query(user_query: str, uploaded_image: Optional[Image.Image], mode: str,
                 include_vector_for_text: bool, use_caption_for_sql: bool):
    if not user_query or not user_query.strip():
        if mode != "New search" or uploaded_image is None:
            return ("Please provide a question or upload an image.", [], "", pd.DataFrame(),
                    pd.DataFrame(engine.session_log), "", "", "", pd.DataFrame(),
                    gr.CheckboxGroup(choices=_artwork_choices(), value=[]))
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


def _proposal_outputs() -> tuple:
    # Open proposals as images (to judge by eye) and as checkboxes 'A12 - RP-P-... (1602) title: reason'
    proposals = engine.pending_proposals()
    gallery = [(p["image"], f"A{p['id']} · {p['object']} · {p['date']}") for p in proposals
               if str(p["image"]).startswith("http")]
    choices = [f"A{p['id']} - {p['object']} ({p['date']}) {str(p['title'])[:40]}: {p['reason']}" for p in proposals]
    return gallery, gr.CheckboxGroup(choices=choices, value=[])


def handle_save_annotation(selected: list, label: str, note: str, grounds: list):
    try:
        status = engine.save_annotation(selected, label, note, grounds)
    except Exception as e:
        status = f"Error: {e}"
    saved = status.startswith("Saved")
    # Clear the fields after a successful save; keep them (to correct) after a warning or error
    return (status, engine.context_df, engine.results.head(MAX_RESULT_ROWS_DISPLAY), pd.DataFrame(engine.session_log),
            "" if saved else label, "" if saved else note, "" if saved else gr.update(), *_proposal_outputs())


def handle_resolve_proposals(selected: list, status: str):
    try:
        message = engine.resolve_proposals(selected, status)
    except Exception as e:
        message = f"Error: {e}"
    return (message, engine.context_df, engine.results.head(MAX_RESULT_ROWS_DISPLAY), pd.DataFrame(engine.session_log),
            *_proposal_outputs())


def handle_draft_annotation(researcher_text: str):
    try:
        draft = engine.draft_annotation(researcher_text)
    except Exception as e:
        draft = {"numbers": [], "label": "", "note": "", "message": f"Error: {e}"}
    # Tick the drafted artworks (the values must match the checkbox labels exactly)
    choices = _artwork_choices()
    selected = [c for c in choices if int(re.match(r"Artwork #(\d+)", c).group(1)) in draft["numbers"]]
    return (gr.CheckboxGroup(choices=choices, value=selected), draft["label"], draft["note"], draft["message"],
            pd.DataFrame(engine.session_log))


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
        Insights you add are saved separately and used in every later search that retrieves those records.
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

            # Researcher insights: saved in the annotations database, shown in the tables and given to the LLM
            with gr.Accordion("Add an insight about the artworks in the context", open=True):
                # Optional: describe the insight freely and let the LLM fill in the fields below (nothing is saved
                # until you click Save)
                annotation_input = gr.Textbox(label="Your insight in your own words (optional)", lines=2,
                                              placeholder="e.g. #2 is a later state of #1, the address was "
                                                          "changed")
                draft_annotation_btn = gr.Button("Draft with AI")
                annotation_artworks = gr.CheckboxGroup(label="Artworks", choices=[])
                annotation_label = gr.Textbox(label="Label (short)", placeholder="e.g. later state")
                annotation_note = gr.Textbox(label="Note", lines=3,
                                             placeholder="e.g. A later state of the same plate, with a changed "
                                                         "publisher's address.")
                # Not every insight travels the same way: choose on which grounds the system may propose other prints
                annotation_grounds = gr.CheckboxGroup(
                    label="Propose other prints this insight may apply to, by",
                    choices=PROPAGATION_GROUNDS, value=PROPAGATION_DEFAULT_GROUNDS,
                    info=("e.g. a copy follows the design, an influence looks similar, a reissue may follow the date; "
                          "untick all for no proposals"),
                )
                save_annotation_btn = gr.Button("Save insight")
                annotation_status = gr.Markdown()

                # The system's contribution: other prints a saved insight may also apply to. Nothing counts as
                # knowledge (or reaches the LLM) until you confirm it; rejected proposals are never proposed again
                gr.Markdown("**Proposed extensions** (suggested by the system). Confirm what you checked and agree "
                            "with; reject only what you checked and found wrong. Leave unsure ones unticked: they "
                            "stay open for later.")
                proposal_gallery = gr.Gallery(label="Proposed prints", columns=4, height=220)
                proposal_choices = gr.CheckboxGroup(label="Proposals", choices=[])
                with gr.Row():
                    confirm_btn = gr.Button("Confirm selected (insight applies)")
                    reject_btn = gr.Button("Reject selected (does not apply, never proposed again)")
                proposal_status = gr.Markdown()

            all_matches_table = gr.Dataframe(label="All matches", wrap=True)
            vlm_box = gr.Textbox(label="VLM description of uploaded image", lines=3, interactive=False)
            with gr.Accordion("Retrieval trace (debug)", open=False):
                trace_box = gr.Textbox(label="Step-by-step trace", lines=16, interactive=False)
            session_log_table = gr.Dataframe(label="Session log", wrap=True)

    all_outputs = [output_answer, output_gallery, sql_box, output_table, session_log_table, trace_box, vlm_box,
                   profile_box, all_matches_table, annotation_artworks]
    submit_btn.click(
        fn=handle_query,
        inputs=[user_input, image_input, mode_toggle, vector_for_text_checkbox, caption_for_sql_checkbox],
        outputs=all_outputs,
    )
    rerun_btn.click(fn=handle_rerun_sql, inputs=[sql_box], outputs=all_outputs)
    save_annotation_btn.click(
        fn=handle_save_annotation,
        inputs=[annotation_artworks, annotation_label, annotation_note, annotation_grounds],
        outputs=[annotation_status, output_table, all_matches_table, session_log_table, annotation_label,
                 annotation_note, annotation_input, proposal_gallery, proposal_choices],
    )
    proposal_outputs = [proposal_status, output_table, all_matches_table, session_log_table, proposal_gallery,
                        proposal_choices]
    confirm_btn.click(fn=lambda sel: handle_resolve_proposals(sel, "confirmed"), inputs=[proposal_choices],
                      outputs=proposal_outputs)
    reject_btn.click(fn=lambda sel: handle_resolve_proposals(sel, "rejected"), inputs=[proposal_choices],
                     outputs=proposal_outputs)
    # Open proposals from earlier sessions are shown as soon as the app opens
    demo.load(fn=_proposal_outputs, outputs=[proposal_gallery, proposal_choices])
    draft_annotation_btn.click(
        fn=handle_draft_annotation,
        inputs=[annotation_input],
        outputs=[annotation_artworks, annotation_label, annotation_note, annotation_status, session_log_table],
    )
    export_results_btn.click(fn=engine.export_results_csv, outputs=[export_results_file])
    export_btn.click(fn=engine.export_log_csv, outputs=[export_file])

if __name__ == "__main__":
    # share=True for Colab (has no accessible localhost)
    demo.queue().launch(share=False, debug=False)