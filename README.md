*Rijksmuseum Graphics Arts AI Assistant - A hybrid Text-to-SQL and Multimodal Vector Search AI system with Gradio UI.*
Works with the first CSV file of the Rijkmuseum ('202020-rma-csv-collection.zip') downloadable on the site of the Rijksmuseum (https://data.rijksmuseum.nl/docs/data-dumps/historical-dumps).
After importing the CSV, run the preprocessor.py script to create the SQL database; now the main.py script can be run.
The model only works on the early modern prints and drawings collection - only these images are embedded (roughly 200 000). These index files are already created and must be imported locally!
