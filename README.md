# Rijksmuseum Graphic Arts Research Assistant

A research prototype for exploring the Rijksmuseum's early modern prints and drawings (1450–1850) with natural
language and images: text-to-SQL and visual similarity search, answers that cite their sources, and a layer of
researcher insights that the system can propose to extend across the collection. Gradio interface.

## Data

The first CSV file (the artworks dataset) of `202020-rma-csv-collection.zip`, from the Rijksmuseum's
[historical data dumps](https://data.rijksmuseum.nl/docs/data-dumps/historical-dumps). The prototype works on the
subcollection of early modern prints and drawings (about 250,000 of 600,000 records); about 194,000 of these have
an image that can be searched visually.

## Setup

1. Download and unzip the CSV.
2. Run `preprocessor.py` to create the SQLite database.
3. Build the visual index with `build_faiss_index_colab.ipynb` (on a GPU). Embedding the images is slow; lower
   the maximum number of images if needed. The script only loads an index that matches its database.
4. Run `rma_assistant.py` and open the Gradio link.

## Models

All tests were run with `Qwen/Qwen2.5-7B-Instruct` (4-bit, about 5–6 GB GPU memory); set it in section 0 of the
script. The default 1.5B model is fine for trying out the interface, but its answers are clearly less reliable.
Images are embedded with CLIP; an optional small vision model describes uploaded images.

## Files written

`annotations.db` (the researcher's insights, the only file the app writes to during a session; on Colab keep it
on Drive), `session_log.csv` and `search_results.csv` (exports).

## Documentation

`rma_assistant_documentation.md` describes how the system works, its limitations, and what would change for
another dataset (section 3.1). The prototype is deliberately specific to this collection.



<img width="1885" height="1078" alt="Schermafbeelding 2026-10-01 145659" src="https://github.com/user-attachments/assets/c6833378-9952-4d2b-af41-1db3b63dd888" />

<img width="1893" height="1073" alt="Schermafbeelding 2026-10-01 145752" src="https://github.com/user-attachments/assets/ed4fcb2f-3cef-4898-9ef7-7f5bb2c551e2" />
