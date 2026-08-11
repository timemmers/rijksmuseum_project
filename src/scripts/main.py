"""
Rijksmuseum Artwork AI Assistant - A hybrid Text-to-SQL and Multimodal Vector Search AI system with Gradio UI.
---
python -m pip install torch transformers faiss-cpu accelerate numpy requests gradio pillow sentence-transformers torchvision pandas lxml qwen-vl-utils
"""

from pathlib import Path
import re

import sqlite3
import json
import io

import requests
from PIL import Image
import numpy as np
import faiss

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline, \
    Qwen2_5_VLForConditionalGeneration, AutoProcessor
import gradio as gr

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


# 1. IMPORT PRESETS

from presets import generation_presets
normalized_presets = {key.lower(): value for key, value in generation_presets.items()}


# 2. DATABASE SETUP & SCHEMA EXTRACTION

current_directory = Path(__file__).parent
database_path = current_directory.parent / "preprocessing" / "rma_artworks"


def connect_and_get_schema(db_path: Path):
    """
    Connects to SQLite database, dynamically detects existing tables,
    and constructs a readable schema string for Text-to-SQL prompting.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]

    if not tables:
        print(f"[Database Warning] No tables found in {db_path}. Creating sample table.")
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS artworks
                       (
                           objectInventoryNumber
                           TEXT
                           PRIMARY
                           KEY,
                           objectPersistentIdentifier
                           TEXT,
                           objectTitle
                           TEXT,
                           objectType
                           TEXT,
                           objectCreator
                           TEXT,
                           objectCreationDate
                           TEXT,
                           objectImage
                           TEXT
                       )
                       """)
        conn.commit()
        tables = ["artworks"]

    main_table = tables[0]
    db_schema = {}

    for table_name in tables:
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        columns = [f"{col_info[1]} ({col_info[2]})" for col_info in cursor.fetchall()]
        db_schema[table_name] = columns

    schema_str = ""
    for table_name, columns in db_schema.items():
        schema_str += f"Table: {table_name}\nColumns: {', '.join(columns)}\n\n"

    print(f"[Database] Connected to '{db_path}'. Primary table detected: '{main_table}'")
    return conn, main_table, schema_str


conn, main_table_name, schema_str = connect_and_get_schema(database_path)

# SYSTEM INSTRUCTS
SQL_FILTER_SYS = (
    "You are an AI assistant for the Rijksmuseum cultural heritage database. "
    "Your task is to convert natural language queries into valid SQL queries based strictly on the database schema below.\n"
    f"Database schema:\n{schema_str}\n"
    "Return ONLY the executable SQL query string without explanation, markdown code blocks, backticks, or array indices like [1]."
)

SYNTHESIS_SYS = (
    "You are an expert art historian for the Rijksmuseum. "
    "Your task is to answer the user's question accurately using ONLY the provided artwork database context and visual analysis below.\n"
    "Guidelines:\n"
    "- Synthesize metadata and image observations into a well-structured, natural response.\n"
    "- Do NOT invent facts outside of the provided context.\n"
    "- If no matching records were found, clearly state that no records matched the query."
)


# 3. AUTOMATIC IMAGE EMBEDDING & FAISS INDEXING

clip_model_name = "sentence-transformers/clip-ViT-B-32"
print(f"[Embeddings] Loading CLIP model ({clip_model_name})...")
embedding_model = SentenceTransformer(clip_model_name)


def build_or_load_faiss_index(db_conn: sqlite3.Connection, table_name: str, embed_model: SentenceTransformer,
                              max_images: int = 300):
    """
    Checks if FAISS vector index exists. If missing, automatically extracts image URLs
    from SQLite (up to max_images), downloads images, and builds the FAISS index using GPU if available.
    """
    index_file = current_directory / "artworks.index"
    row_ids_file = current_directory / "row_ids.npy"

    # Check if index already exists
    if index_file.exists() and row_ids_file.exists():
        print(f"[FAISS] Found existing index files: '{index_file.name}' and '{row_ids_file.name}'. Loading...")
        faiss_index = faiss.read_index(str(index_file))
        row_ids = np.load(str(row_ids_file))
        return faiss_index, row_ids

    print(f"[FAISS] Starting automatic image embedding (Limited to max {max_images} images for testing)...")
    cursor = db_conn.cursor()

    # Detect image column
    cursor.execute(f"PRAGMA table_info('{table_name}');")
    columns = [col_info[1] for col_info in cursor.fetchall()]
    image_col = next((col for col in columns if "image" in col.lower() or "url" in col.lower()), None)

    if not image_col:
        print("[FAISS Warning] No image column detected. Initializing empty FAISS index.")
        return faiss.IndexFlatIP(512), np.array([], dtype=int)

    # Added LIMIT clause to fetch only the specified number of records
    query = f"""
        SELECT rowid, {image_col} 
        FROM {table_name} 
        WHERE {image_col} IS NOT NULL AND {image_col} != '<null>' AND {image_col} != '' 
        LIMIT {max_images}
    """
    cursor.execute(query)
    records = cursor.fetchall()

    print(f"[FAISS] Found {len(records)} image records to process.")

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model.to(device)
    print(f"[FAISS] Running embedding model on device: {device.upper()}")

    embeddings = []
    valid_row_ids = []
    headers = {'User-Agent': 'Mozilla/5.0'}

    for row_id, img_url in records:
        try:
            if not isinstance(img_url, str) or not img_url.startswith("http"):
                continue

            res = requests.get(img_url, timeout=5, headers=headers)
            if res.status_code == 200:
                img = Image.open(io.BytesIO(res.content)).convert("RGB")
                vec = embed_model.encode(img, convert_to_numpy=True)
                embeddings.append(vec)
                valid_row_ids.append(row_id)
                print(f" -> Successfully embedded rowid: {row_id}")
        except Exception as err:
            print(f" -> Skipping rowid {row_id} due to error: {err}")

    if len(embeddings) > 0:
        embeddings_np = np.array(embeddings, dtype='float32')
        faiss.normalize_L2(embeddings_np)

        dimension = embeddings_np.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(embeddings_np)
        row_ids_np = np.array(valid_row_ids, dtype=int)

        faiss.write_index(faiss_index, str(index_file))
        np.save(str(row_ids_file), row_ids_np)
        print(f"[FAISS] Successfully created index with {len(valid_row_ids)} image vectors!")
        return faiss_index, row_ids_np
    else:
        print("[FAISS Warning] No valid images were embedded. Initializing empty index.")
        return faiss.IndexFlatIP(512), np.array([], dtype=int)


faiss_index, row_ids = build_or_load_faiss_index(conn, main_table_name, embedding_model, max_images=300)


# 4. RECIPROCAL RANK FUSION (RRF)

def reciprocal_rank_fusion(sql_ids: list, vector_ids: list, k: int = 60, top_n: int = 7) -> list:
    """
    Combines ranked lists from SQL and Vector searches using Reciprocal Rank Fusion.
    Score(d) = sum(1 / (k + rank_i))
    """
    scores = {}

    for rank, doc_id in enumerate(sql_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    sorted_ids = sorted(scores.keys(), key=lambda item: scores[item], reverse=True)
    return sorted_ids[:top_n]


# 5. LANGUAGE MODEL INITIALIZATION

print("[LLM] Loading text synthesis model (Qwen/Qwen2.5-1.5B-Instruct)...")
llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa" if torch.cuda.is_available() else "eager"
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

vlm_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
vlm_model = None
vlm_processor = None

try:
    print(f"[VLM] Attempting to load vision model ({vlm_model_id})...")
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_id)
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    print("[VLM] Vision model loaded successfully!")
except Exception as e:
    print(f"[VLM Notice] Could not initialize Qwen-VL model: {e}")
    print("[VLM Notice] Ensure 'transformers', 'torchvision', and 'qwen-vl-utils' are installed.")


def generate(messages_or_sys, config_or_user=None, max_new_tokens=500, temperature=0.0, **kwargs):
    if isinstance(messages_or_sys, str):
        messages = [
            {"role": "system", "content": messages_or_sys},
            {"role": "user", "content": config_or_user}
        ]
        cfg_temp = temperature
        cfg_max = max_new_tokens
        cfg_rep = kwargs.get("repetition_penalty", 1.05)
    else:
        messages = messages_or_sys
        cfg_temp = config_or_user.get("temperature", 0.0)
        cfg_max = config_or_user.get("max_new_tokens", 500)
        cfg_rep = config_or_user.get("repetition_penalty", 1.1)

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    device = next(llm_model.parameters()).device
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    do_sample = cfg_temp > 0.0
    gen_cfg = GenerationConfig(
        do_sample=do_sample,
        max_new_tokens=cfg_max,
        repetition_penalty=cfg_rep,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if do_sample:
        gen_cfg.temperature = cfg_temp
        if isinstance(messages_or_sys, list):
            gen_cfg.top_p = config_or_user.get("top_p", 1.0)
            gen_cfg.top_k = config_or_user.get("top_k", 50)
        else:
            gen_cfg.top_p = kwargs.get("top_p", 1.0)
            gen_cfg.top_k = kwargs.get("top_k", 50)

    output_ids = llm_model.generate(input_ids, attention_mask=attention_mask, generation_config=gen_cfg)
    generated_ids = output_ids[:, input_ids.shape[1]:]

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


# 6. MODEL ENGINE CLASS

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
        for key, value in presets.items():
            self.config[key] = value

    def _execute_sql_search(self, user_input: str) -> list:
        try:
            raw_sql = generate(
                SQL_FILTER_SYS,
                f"Question: {user_input}\nSQL query:",
                max_new_tokens=250,
                temperature=0.0
            )

            clean_sql = re.sub(r"```sql\s*", "", raw_sql, flags=re.IGNORECASE)
            clean_sql = clean_sql.replace("```", "").strip()
            clean_sql = re.sub(r"^(sql query:|sql:)\s*", "", clean_sql, flags=re.IGNORECASE).strip()

            clean_sql = re.sub(r'\[\d+\]', '', clean_sql)

            print(f"[SQL Query] Generated: {clean_sql}")

            dangerous_keywords = ["drop", "delete", "update", "insert", "alter", "create", "truncate"]
            if any(keyword in clean_sql.lower() for keyword in dangerous_keywords):
                print("[SQL Security] Dangerous keyword detected. Aborting SQL execution.")
                return []

            if re.search(r"(?i)\bFROM\b", clean_sql):
                sql_for_ids = re.sub(r"(?i)^SELECT\s+.*?\s+FROM\s+\S+", f"SELECT rowid FROM {self.table_name}",
                                     clean_sql)
            else:
                sql_for_ids = f"SELECT rowid FROM {self.table_name} WHERE {clean_sql}"

            cursor = self.db_conn.cursor()
            cursor.execute(sql_for_ids)
            results = cursor.fetchall()

            sql_row_ids = [int(row[0]) for row in results if row[0] is not None]
            print(f"[SQL Results] Found {len(sql_row_ids)} row IDs.")
            return sql_row_ids

        except Exception as e:
            print(f"[SQL Error] Query failed: {e}. Defaulting to empty SQL list.")
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

    def process_query(self, user_input: str, user_image: Image.Image = None) -> dict:
        try:
            sql_ids = self._execute_sql_search(user_input) if user_input and user_input.strip() else []
            vector_ids = self._execute_vector_search(user_input=user_input, user_image=user_image, k=15)

            hybrid_ids = reciprocal_rank_fusion(sql_ids, vector_ids, k=60, top_n=self.retrieval_k)
            print(f"[Hybrid RRF Results] Top ranked row IDs: {hybrid_ids}")

            image_analysis_text = ""
            if user_image is not None and vlm_model is not None and vlm_processor is not None:
                print("[VLM] Analyzing uploaded image...")
                try:
                    vlm_messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": user_image},
                                {"type": "text",
                                 "text": "Describe the key visual details, artistic style, medium, and subject matter of this image."}
                            ]
                        }
                    ]

                    # Process text and image inputs correctly for Qwen2.5-VL using process_vision_info
                    prompt_text = vlm_processor.apply_chat_template(vlm_messages, tokenize=False,
                                                                    add_generation_prompt=True)

                    if process_vision_info is not None:
                        image_inputs, video_inputs = process_vision_info(vlm_messages)
                        inputs = vlm_processor(
                            text=[prompt_text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt"
                        )
                    else:
                        inputs = vlm_processor(
                            text=[prompt_text],
                            images=[user_image],
                            padding=True,
                            return_tensors="pt"
                        )

                    inputs = inputs.to(vlm_model.device)

                    with torch.no_grad():
                        generated_ids = vlm_model.generate(**inputs, max_new_tokens=300)

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    image_analysis_text = vlm_processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    print(f"[VLM Analysis Complete]: {image_analysis_text[:100]}...")
                except Exception as vlm_err:
                    print(f"[VLM Error] Image analysis failed: {vlm_err}")
                    image_analysis_text = ""

            if not hybrid_ids:
                if image_analysis_text:
                    return {
                        "answer": f"Visual Analysis of Uploaded Image:\n{image_analysis_text}\n\n(No matching database records found.)",
                        "images": []
                    }
                return {"answer": "No artworks match the specified query or image.", "images": []}

            cursor = self.db_conn.cursor()
            placeholders = ",".join("?" for _ in hybrid_ids)
            cursor.execute(f"SELECT rowid, * FROM {self.table_name} WHERE rowid IN ({placeholders})", hybrid_ids)

            raw_rows = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description][1:]

            row_dict = {row[0]: row[1:] for row in raw_rows}
            ordered_rows = [row_dict[rid] for rid in hybrid_ids if rid in row_dict]

            context_parts = []
            matching_image_urls = []

            for row in ordered_rows:
                row_details = [f"{col}: {val}" for col, val in zip(column_names, row) if val and str(val).strip()]
                context_parts.append(" | ".join(row_details))

                for col_name, val in zip(column_names, row):
                    if ("image" in col_name.lower() or "url" in col_name.lower()) and val and isinstance(val,
                                                                                                         str) and val.startswith(
                            "http"):
                        matching_image_urls.append(val)

            context_str = "\n\n".join(context_parts)
            user_prompt = f"Database Context:\n{context_str}\n\n"
            if image_analysis_text:
                user_prompt += f"Visual Analysis of Uploaded Image:\n{image_analysis_text}\n\n"
            user_prompt += f"Question: {user_input}"

            answer_text = generate(SYNTHESIS_SYS, user_prompt, **self.config)

            return {"answer": answer_text, "images": matching_image_urls}

        except Exception as e:
            return {"answer": f"Error during processing: {e}", "images": []}


engine = ModelEngine(db_conn=conn, table_name=main_table_name, faiss_idx=faiss_index, row_ids_arr=row_ids)


# 7. GRADIO FRONTEND INTERFACE

def response_handler(user_query: str, uploaded_image: Image.Image, preset_name: str):
    if (not user_query or not user_query.strip()) and uploaded_image is None:
        return "Please provide a natural language prompt or upload an image.", []

    choice = preset_name.lower()
    if choice in normalized_presets:
        engine.apply_preset(normalized_presets[choice])

    output = engine.process_query(user_input=user_query, user_image=uploaded_image)
    return output.get("answer", "No response generated."), output.get("images", [])


available_presets = list(generation_presets.keys())

with gr.Blocks(title="Rijksmuseum Collection Assistant") as demo:
    gr.Markdown(
        """
        # Rijksmuseum Collection Assistant
        Search and explore the Rijksmuseum collection using natural language and images.
        The model combines automatic Text-to-SQL with Multimodal Vector Search. Reciprocal Rank Fusion (RRF) filters the retrieved results for maximum relevance.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            user_input = gr.Textbox(
                label="User Question",
                placeholder="e.g. Which 17th century prints depict landscapes or candles?",
                lines=3
            )
            image_input = gr.Image(
                label="Upload Image (Optional)",
                type="pil"
            )
            preset_dropdown = gr.Dropdown(
                choices=available_presets,
                value="balanced",
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
    demo.launch(share=False)