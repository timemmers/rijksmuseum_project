from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import sqlite3


# SET THE DIRECTORIES UP

current_directory = Path(__file__).parent

csv_file = current_directory.parent / "data" / "rma_artworks.csv" # input
database = current_directory.parent / "rma_artworks" # output


# CREATE SQLITE DATABASE WITH PANDAS

def csv_to_sql():
    df = pd.read_csv(csv_file)

    conn = sqlite3.connect("rma_artworks")

    df.to_sql("rma_artworks", conn, if_exists="replace", index=False)
    conn.close()


##########################

if __name__ == "__main__":
    csv_to_sql()

