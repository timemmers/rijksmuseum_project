"""
Rijksmuseum Graphics Arts AI Assistant - A hybrid Text-to-SQL and Multimodal Vector Search AI system with Gradio UI.
---
python -m pip install torch transformers faiss-cpu accelerate numpy requests gradio pillow sentence-transformers pandas
"""

from pathlib import Path
import re
import sqlite3
import io
from typing import Optional

import requests
from PIL import Image
import numpy as np
import faiss

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoProcessor, AutoModelForImageTextToText
import gradio as gr
import gc


# 1. IMPORT PRESETS

from presets import generation_presets
normalized_presets = {key.lower(): value for key, value in generation_presets.items()}


# 2. DATABASE SETUP & SCHEMA EXTRACTION

current_directory = Path(__file__).parent
database_path = current_directory.parent / "preprocessing" / "rma_artworks"

def connect_and_get_schema(db_path: Path):
    """
    Connects to SQLite database, dynamically detects existing tables,
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

conn, main_table_name, schema_str = connect_and_get_schema(database_path)

# SYSTEM INSTRUCTS FOR TEXT-TO-SQL
SQL_FILTER_SYS = f"""You are an expert AI assistant that translates natural language questions into executable SQLite SQL queries.
Database schema:
{schema_str}

CRITICAL RULES FOR SQL GENERATION:

1. Scope Limitation: The image database ONLY contains early modern prints and drawings. Always filter by objectType[1] and objectCreationDate[1].
2. Exact Column Names: You MUST use the exact column names with brackets from the schema, such as "objectType[1]", "objectCreator[1]", "objectCreationDate[1]".
3. Dutch Terminology: The database uses Dutch terms!
   - You MUST translate English search terms into Dutch keywords for SQL LIKE clauses.
   - Examples: 'print' -> 'prent', 'drawing' -> 'tekening', 'landscape' -> 'landschap', 'portrait' -> 'portret'.
4. Person Names (Comma Handling): Creator names are stored as 'Lastname, Firstname' (e.g., 'Cort, Cornelis').
   - NEVER match full names directly like: WHERE "objectCreator[1]" LIKE '%Cornelis Cort%'
   - ALWAYS split full names into separate AND words: WHERE "objectCreator[1]" LIKE '%Cornelis%' AND "objectCreator[1]" LIKE '%Cort%'

FEW-SHOT EXAMPLES:

Question: Which prints were produced by Cornelis Cort?
SQL: SELECT rowid FROM {main_table_name} WHERE "objectType[1]" LIKE '%prent%' AND "objectCreator[1]" LIKE '%Cornelis%' AND "objectCreator[1]" LIKE '%Cort%' AND "objectCreationDate[1]" BETWEEN '1450' AND '1850'

Question: Show me drawings of landscapes from the 16th century
SQL: SELECT rowid FROM {main_table_name} WHERE "objectType[1]" LIKE '%tekening%' AND "objectTitle[1]" LIKE '%landschap%' AND "objectCreationDate[1]" BETWEEN '1501' AND '1600'

Question: Show me 17th century portraits
SQL: SELECT rowid FROM {main_table_name} WHERE "objectType[1]" LIKE '%prent%' AND "objectTitle[1]" LIKE '%portret%' AND "objectCreationDate[1]" BETWEEN '1601' AND '1700'

Return ONLY the raw SQL query."""

SYNTHESIS_SYS = (
    "You are an expert art historian for the Rijksmuseum.\n"
    "Your task is to answer the user's question accurately using ONLY the provided artwork database context below.\n"
    "Guidelines:\n"
    "- You MUST explicitly discuss and describe EVERY retrieved artwork provided in the context, in the exact order presented (Artwork #1, Artwork #2, etc.).\n"
    "- If visual analysis of an uploaded image is provided, incorporate those visual details into your artwork analysis.\n"
    "- Do NOT skip any retrieved artwork record from the context.\n"
    "- Do NOT invent facts outside of the provided context.\n"
    "- If no matching records were found, clearly state that no records matched the query."
)


# 3. AUTOMATIC IMAGE EMBEDDING & FAISS INDEXING

clip_model_name = "sentence-transformers/clip-ViT-B-32"
print(f"[Embeddings] Loading CLIP model ({clip_model_name})...")
embedding_model = SentenceTransformer(clip_model_name)

def build_or_load_faiss_index(db_conn: sqlite3.Connection, table_name: str, embed_model: SentenceTransformer,
                              max_images: int = 5000):
    """
    Checks if FAISS vector index exists. If missing, automatically extracts image URLs from early modern prints
    and drawings (up to max_images), downloads images, and builds the FAISS index using GPU if available
    """
    index_file = current_directory / "artworks.index"
    row_ids_file = current_directory / "row_ids.npy"

    if index_file.exists() and row_ids_file.exists():
        print(f"[FAISS] Found existing index files: '{index_file.name}' and '{row_ids_file.name}'. Loading...")
        return faiss.read_index(str(index_file)), np.load(str(row_ids_file))

    print(f"[FAISS] Starting automatic image embedding for early modern prints/drawings (Limited to max {max_images} images for testing)...")
    cursor = db_conn.cursor()

    cursor.execute(f"PRAGMA table_info('{table_name}');")
    columns = [col_info[1] for col_info in cursor.fetchall()]
    image_col = next((col for col in columns if "image" in col.lower() or "url" in col.lower()), None)

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

    for row_id, img_url in records:
        try:
            if isinstance(img_url, str) and img_url.startswith("http"):
                res = requests.get(img_url, timeout=5, headers=headers)
                if res.status_code == 200:
                    img = Image.open(io.BytesIO(res.content)).convert("RGB")
                    embeddings.append(embed_model.encode(img, convert_to_numpy=True))
                    valid_row_ids.append(row_id)
                    print(f" -> Successfully embedded rowid: {row_id}")
        except Exception as err:
            print(f" -> Skipping rowid {row_id} due to error: {err}")

    if embeddings:
        embeddings_np = np.array(embeddings, dtype='float32')
        faiss.normalize_L2(embeddings_np)

        faiss_index = faiss.IndexFlatIP(embeddings_np.shape[1])
        faiss_index.add(embeddings_np)
        row_ids_np = np.array(valid_row_ids, dtype=int)

        faiss.write_index(faiss_index, str(index_file))
        np.save(str(row_ids_file), row_ids_np)
        print(f"[FAISS] Successfully created index with {len(valid_row_ids)} image vectors!")
        return faiss_index, row_ids_np

    print("[FAISS Warning] No valid images were embedded. Initializing empty index.")
    return faiss.IndexFlatIP(512), np.array([], dtype=int)

faiss_index, row_ids = build_or_load_faiss_index(conn, main_table_name, embedding_model, max_images=5000)


# 4. RECIPROCAL RANK FUSION (RRF)

def reciprocal_rank_fusion(sql_ids: list, vector_ids: list, k: int = 60, top_n: int = 7) -> list:
    """
    Combines ranked lists from SQL and Vector searches using Reciprocal Rank Fusion
    Score(d) = sum(1 / (k + rank_i))
    """
    scores = {}

    for rank, doc_id in enumerate(sql_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    return sorted(scores.keys(), key=lambda item: scores[item], reverse=True)[:top_n]


# 5. LLM + VLM INITIALIZATION

llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"[LLM] Loading text synthesis model ({llm_model_name})...")
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"[LLM] ({llm_model_name}) loaded succesfully!")


vlm_model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
vlm_processor = None
vlm_model = None

try:
    print(f"[VLM] Loading Vision Language Model ({vlm_model_id})...")
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_id)
    vlm_model = AutoModelForImageTextToText.from_pretrained(
        vlm_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    print(f"[VLM] {vlm_model_id} loaded successfully!")
except Exception as vlm_err:
    print(f"[VLM Warning] Could not load VLM: {vlm_err}")


def generate(messages_or_sys, config_or_user=None, max_new_tokens=500, temperature=0.0, **kwargs):
    if isinstance(messages_or_sys, str):
        messages = [
            {"role": "system", "content": messages_or_sys},
            {"role": "user", "content": config_or_user}
        ]
        cfg = {"temperature": temperature, "max_new_tokens": max_new_tokens,
               "repetition_penalty": kwargs.get("repetition_penalty", 1.05),
               "top_p": kwargs.get("top_p", 1.0), "top_k": kwargs.get("top_k", 50)}
    else:
        messages = messages_or_sys
        cfg = config_or_user or {}

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length)

    device = next(llm_model.parameters()).device
    input_ids = inputs.input_ids.to(device)

    do_sample = cfg.get("temperature", 0.0) > 0.0
    gen_cfg = GenerationConfig(
        do_sample=do_sample,
        max_new_tokens=cfg.get("max_new_tokens", 500),
        repetition_penalty=cfg.get("repetition_penalty", 1.1),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if do_sample:
        gen_cfg.temperature = cfg.get("temperature")
        gen_cfg.top_p = cfg.get("top_p", 1.0)
        gen_cfg.top_k = cfg.get("top_k", 50)

    output_ids = llm_model.generate(input_ids, attention_mask=inputs.attention_mask.to(device), generation_config=gen_cfg)
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


# 6. MODEL ENGINE

class ModelEngine:
    def __init__(self, db_conn: sqlite3.Connection, table_name: str, faiss_idx, row_ids_arr):
        self.db_conn = db_conn
        self.table_name = table_name
        self.retrieval_k = 7
        self.config = {
            "temperature": 0.5,
            "top_p": 0.85,
            "top_k": 40,
            "max_new_tokens": 1200,
            "repetition_penalty": 1.1
        }
        self.embed_model = embedding_model
        self.faiss_index = faiss_idx
        self.row_ids = row_ids_arr

    def apply_preset(self, presets: dict):
        self.config.update(presets)

    def _analyze_image(self, user_image: Image.Image) -> str:
        """
        Analyzes an uploaded image using the VLM to extract visual features and subject matter
        """
        if user_image is None or vlm_model is None or vlm_processor is None:
            return ""

        try:
            print("[VLM] Generating visual analysis for uploaded image...")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text",
                         "text": "Describe the main visual subject matter, composition, and key artistic details of this artwork."}
                    ]
                }
            ]
            prompt = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = vlm_processor(text=prompt, images=[user_image], return_tensors="pt")

            device = next(vlm_model.parameters()).device
            inputs = inputs.to(device)

            with torch.no_grad():
                generated_ids = vlm_model.generate(**inputs, max_new_tokens=200)

            generated_texts = vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)

            raw_output = generated_texts[0]
            if "Assistant:" in raw_output:
                raw_output = raw_output.split("Assistant:")[-1].strip()

            print(f"[VLM Result]: {raw_output[:100]}...")
            return raw_output.strip()

        except Exception as e:
            print(f"[VLM Error] Image analysis failed: {e}")
            return ""

    def _execute_sql_search(self, user_input: str) -> list:
        try:
            raw_sql = generate(SQL_FILTER_SYS, f"Question: {user_input}\nSQL query:", max_new_tokens=250, temperature=0.0)

            clean_sql = re.sub(r"```sql\s*|```|^(sql query:|sql:)\s*", "", raw_sql, flags=re.IGNORECASE).strip()
            print(f"[SQL Query]: {clean_sql}")

            dangerous_keywords = ["drop", "delete", "update", "insert", "alter", "create", "truncate"]
            if any(keyword in clean_sql.lower() for keyword in dangerous_keywords):
                print("[SQL] Dangerous keyword detected. Aborted.")
                return []

            if re.search(r"(?i)\bFROM\b", clean_sql):
                sql_for_ids = re.sub(r"(?i)^SELECT\s+.*?\s+FROM\s+\S+", f"SELECT rowid FROM {self.table_name}", clean_sql)
            else:
                sql_for_ids = f"SELECT rowid FROM {self.table_name} WHERE {clean_sql}"

            cursor = self.db_conn.cursor()
            cursor.execute(sql_for_ids)
            sql_row_ids = [int(row[0]) for row in cursor.fetchall() if row[0] is not None]
            print(f"[SQL Results] found {len(sql_row_ids)} record IDs.")
            return sql_row_ids

        except Exception as e:
            print(f"[SQL Error] Query failed: {e}.")
            return []

    def _execute_vector_search(self, user_input: str = None, user_image: Image.Image = None, k: int = 15) -> list:
        try:
            if self.faiss_index.ntotal == 0 or len(self.row_ids) == 0:
                print("[Vector Search] FAISS index is empty.")
                return []

            if user_image is not None:
                query_vector = self.embed_model.encode(user_image).astype('float32')
            elif user_input:
                query_vector = self.embed_model.encode(user_input).astype('float32')
            else:
                return []

            if query_vector.ndim == 1:
                query_vector = np.expand_dims(query_vector, axis=0)

            faiss.normalize_L2(query_vector)
            distances, indices = self.faiss_index.search(query_vector, k)

            vector_row_ids = [int(self.row_ids[idx]) for idx in indices[0] if idx != -1 and idx < len(self.row_ids)]
            print(f"[Vector Search Results] Found {len(vector_row_ids)} row IDs.")
            return vector_row_ids

        except Exception as e:
            print(f"[Vector Search Error] {e}")
            return []

    def _fetch_and_format_context(self, hybrid_ids: list):
        """
        Fetch rows from database and construct context and image list
        """
        if not hybrid_ids:
            return "", []

        cursor = self.db_conn.cursor()
        placeholders = ",".join("?" for _ in hybrid_ids)
        cursor.execute(f"SELECT rowid, * FROM {self.table_name} WHERE rowid IN ({placeholders})", hybrid_ids)

        raw_rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description][1:]

        row_dict = {row[0]: row[1:] for row in raw_rows}

        context_parts = []
        matching_image_urls = []

        for index, rid in enumerate(hybrid_ids, start=1):
            if rid not in row_dict:
                continue

            row = row_dict[rid]
            row_details = [f"{col}: {val}" for col, val in zip(column_names, row) if val and str(val).strip()]
            context_parts.append(f"[Artwork #{index} - ID {rid}] " + " | ".join(row_details))

            for col_name, val in zip(column_names, row):
                if ("image" in col_name.lower() or "url" in col_name.lower()) and isinstance(val, str) and val.startswith("http"):
                    matching_image_urls.append(val)
                    break

        return "\n\n".join(context_parts), matching_image_urls

    def process_query(self, user_input: str, user_image: Image.Image = None) -> dict:
        try:
            sql_ids = self._execute_sql_search(user_input) if user_input and user_input.strip() else []
            vector_ids = self._execute_vector_search(user_input=user_input, user_image=user_image, k=15)

            if sql_ids:
                sql_set = set(sql_ids)
                filtered_vector_ids = [vid for vid in vector_ids if vid in sql_set]

                hybrid_ids = reciprocal_rank_fusion(sql_ids, filtered_vector_ids, k=60, top_n=self.retrieval_k)
            else:
                hybrid_ids = reciprocal_rank_fusion(sql_ids, vector_ids, k=60, top_n=self.retrieval_k)

            print(f"[Hybrid RRF Results] Top ranked row IDs: {hybrid_ids}")

            image_analysis = self._analyze_image(user_image) if user_image is not None else ""

            if not hybrid_ids:
                if image_analysis:
                    return {
                        "answer": f"Visual Analysis of Uploaded Image:\n{image_analysis}\n\nNo matching database records found.",
                        "images": []
                    }
                return {"answer": "No artworks match the specified query or uploaded image.", "images": []}

            context_str, matching_image_urls = self._fetch_and_format_context(hybrid_ids)

            effective_query = user_input.strip() if user_input and user_input.strip() else "Describe the retrieved artwork records matching the uploaded image."

            user_prompt = f"Database Context:\n{context_str}\n\n"
            if image_analysis:
                user_prompt += f"Visual Analysis of Uploaded Image:\n{image_analysis}\n\n"

            user_prompt += f"Question: {effective_query}"

            answer_text = generate(SYNTHESIS_SYS, user_prompt, **self.config)
            return {"answer": answer_text, "images": matching_image_urls}

        except Exception as e:
            return {"answer": f"Error during processing: {e}", "images": []}

engine = ModelEngine(db_conn=conn, table_name=main_table_name, faiss_idx=faiss_index, row_ids_arr=row_ids)


# 7. GRADIO FRONTEND INTERFACE

def response_handler(user_query: str, uploaded_image: Optional[Image.Image], preset_name: str):
    if (not user_query or not user_query.strip()) and uploaded_image is None:
        return "Please provide a natural language prompt or upload an image.", []

    try:
        formatted_image = uploaded_image.convert("RGB") if uploaded_image is not None else None

        choice = preset_name.lower()
        if choice in normalized_presets:
            engine.apply_preset(normalized_presets[choice])

        output = engine.process_query(user_input=user_query, user_image=formatted_image)

        answer_text = output.get("answer", "No response generated.")
        matched_images = output.get("images", [])

        return answer_text, matched_images

    except Exception as e:
        print(f"[Response handler error] {e}")
        return f"Error during processing: {e}", []

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


available_presets = list(generation_presets.keys())

with gr.Blocks(title="Rijksmuseum Collection Assistant") as demo:
    gr.Markdown(
        """
        # Rijksmuseum Graphic Arts Assistant
        Search and explore the early modern prints and drawings collection of the Rijksmuseum using natural language and images.
        The model combines automatic Text-to-SQL with Multimodal Vector Search. Reciprocal Rank Fusion (RRF) filters the retrieved results for maximum relevance.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            user_input = gr.Textbox(
                label="User Question",
                placeholder="e.g. Which prints were produced by Cornelis Cort?",
                lines=3
            )

            image_input = gr.Image(
                label="Upload Image (Optional)",
                type="pil"
            )
            preset_dropdown = gr.Dropdown(
                choices=available_presets,
                value="Deterministic",
                label="Decoding Preset",
                info="Choose generation behavior."
            )
            submit_btn = gr.Button("Search and Synthesize", variant="primary")

        with gr.Column(scale=2):
            output_answer = gr.Textbox(
                label="Generated Response",
                lines=12,
                interactive=False
            )
            output_gallery = gr.Gallery(
                label="Matched Artwork Images",
                columns=3,
                height=300
            )

    submit_btn.click(
        fn=response_handler,
        inputs=[user_input, image_input, preset_dropdown],
        outputs=[output_answer, output_gallery]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)