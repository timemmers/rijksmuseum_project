*Rijksmuseum graphics arts AI assistant - A text-to-SQL and multimodal vector search AI system with Gradio UI*
---
*This prototype allows users to interact with the collection of early modern prints and drawings of the Rijksmuseum, using both natural language and images.*


Works with the first CSV file (the artworks dataset) of the Rijkmuseum ('202020-rma-csv-collection.zip') downloadable on the site of the Rijksmuseum (https://data.rijksmuseum.nl/docs/data-dumps/historical-dumps).

After importing the CSV, run the preprocessor.py script to create the SQLite database. After this, the main.py script can be executed.

This model only works on the subcollection (250 000 records of total 600 000) of early modern prints and drawings. Naturally, only the images of this subcollection are embedded and retrievable (roughly 200 000). NOTE: creating the image embeddings is computationally intensive, hence the 'max_images' guardrail. Use a GPU or tone down the max number of images. Upscale the size of the LLMs depending on your computational power. More extensive documentation can be found in the rma_assistant_documentation.md file.

<img width="1885" height="1078" alt="Schermafbeelding 2026-10-01 145659" src="https://github.com/user-attachments/assets/c6833378-9952-4d2b-af41-1db3b63dd888" />

<img width="1893" height="1073" alt="Schermafbeelding 2026-10-01 145752" src="https://github.com/user-attachments/assets/ed4fcb2f-3cef-4898-9ef7-7f5bb2c551e2" />
