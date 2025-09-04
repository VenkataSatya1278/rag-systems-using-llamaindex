# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: training-env
#     language: python
#     name: python3
# ---

# %% [markdown] id="V9G5p8ENYHkP"
# # Data Loaders

# %% [markdown]
# #### Contents:
# Different type of loaders in Llama
# - SimpleDirectoryReader
# - PDFReader
# - GoogleDocsReader
# - MarkDownReader
# - HTMLReader
# - CSVReader
#
# Different Data connectors in Llama
# - Integration
# - Dynamic Data
# - Complex Queries
# Most frequently Data Connectors
#  - SQLdatabaseReader
#  - FirestoreReader
#  - MongoDBReader
#  - APIReader
#
#  Llama Hub -  Repsitory o DataLoaders
#
# 1. Read PDF Files
# 2. Read CSV Files
# 3. Load Webpage
# 4. Reading from Directory

# %% [markdown]
# pip install llama-index-readers-file llama-index-readers-web unstructured

# %%
# !pip install llama-index-readers-file llama-index-readers-web unstructured

# %%
# !pip install wget


# %% [markdown] id="xqRmn0D4YmZV"
# ## 1. Loading PDF files

# %% [markdown]
# Next, we'll download a PDF file from a given URL and save it into our `data` directory. Here, we are using the `wget` command to download the file. The URL points to a PDF file hosted on GitHub, and we save it as `transformers.pdf` in our `data` directory.

# %%
# !mkdir data
# !wget "https://arxiv.org/pdf/1706.03762" -O 'data/transformers.pdf'

# %% [markdown] id="usMi3LSwZwU2"
# ## Using PDFReader

# %%
from pathlib import Path
from llama_index.readers.file import PDFReader

# %% [markdown]
# Next, we create an instance of `PDFReader`.

# %% executionInfo={"elapsed": 498, "status": "ok", "timestamp": 1703166832879, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="dBLPDbZ5u5_D"
loader = PDFReader()

# %% [markdown]
# We then use the `load_data` method to load the content of our PDF file. The `file` parameter specifies the path to our PDF file. The `load_data` method reads the PDF and returns a list of documents, where each document represents a portion of the PDF content.

# %% executionInfo={"elapsed": 4233, "status": "ok", "timestamp": 1703166842594, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="wJAWQF2amw01"
documents = loader.load_data(file=Path('./data/transformers.pdf'))

# %% [markdown]
#
# To check how many documents we have loaded, we can use the `len` function.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1703166844171, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="QEWM0nLn5IF-" outputId="876da4dc-65e1-447e-de40-bcaf79a4da10"
len(documents)

# %% [markdown]
# Finally, we can access the text of the first document in our list and display it. This gives us a peek into the content of the PDF.

# %% colab={"base_uri": "https://localhost:8080/", "height": 143} executionInfo={"elapsed": 445, "status": "ok", "timestamp": 1703166874383, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="tvf0BdaK5JnY" outputId="d3ad55cd-f605-4269-8a43-0c58af52a305"
print(documents[0].text)

# %%
documents[0].to_dict().keys()

# %%
documents[0].id_

# %%
documents[0].metadata

# %% [markdown] id="rE8-nEIJm7rQ"
# ## 2. Loading CSV files

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2686, "status": "ok", "timestamp": 1703166890763, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="JQED6MsYoiOM" outputId="22d50d7e-7691-456d-ba9e-aacbe3cbfeaf"
# !wget https://datahack-prod.s3.amazonaws.com/train_file/train_v9rqX0R.csv -O 'data/transactions.csv'

# %%
from llama_index.readers.file import CSVReader

# %% [markdown]
# Next, we create an instance of CSVReader.

# %% executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1703166897171, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="FWCx0bDWm_ws"
loader = CSVReader()

# %% [markdown]
# We then use the load_data method to load the content of our CSV file. The file parameter specifies the path to our CSV file. This method reads the CSV file and returns a list of documents, each representing a row or a set of rows from the CSV.

# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1703166898799, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="ZpNkUqdJm_ze"
documents = loader.load_data(file=Path('./data/transactions.csv'))

# %% [markdown]
# To access the content of the first document, we simply reference it by its index and display the text.

# %%
len(documents)

# %% colab={"base_uri": "https://localhost:8080/", "height": 143} executionInfo={"elapsed": 1769, "status": "ok", "timestamp": 1703166902028, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="XLIAleMHx_ct" outputId="e5d035ee-7a84-4f7c-de24-7e84543edc42"
documents[0].text

# %% [markdown] id="dkY4prj0ufI-"
# ## 3. Loading Web Page

# %% [markdown]
# We start by importing the UnstructuredURLLoader from llama_index.readers.web. This class helps us load and parse content from web pages.

# %%
from llama_index.readers.web import UnstructuredURLLoader

# %% [markdown]
# We create an instance of UnstructuredURLLoader and pass a list of URLs we want to load.

# %% executionInfo={"elapsed": 439, "status": "ok", "timestamp": 1703166923015, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="mj6lXkdh1lxT"
loader = UnstructuredURLLoader(urls=['https://huggingface.co/blog/moe'])

# %% [markdown]
# Using the load_data method, we load the content of the specified URL.

# %% executionInfo={"elapsed": 398, "status": "ok", "timestamp": 1703166925047, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="-Q0iIWMN1nJg"
documents = loader.load_data()

# %%
len(documents)

# %% [markdown]
# We can access and display the text of the first document similarly.

# %% colab={"base_uri": "https://localhost:8080/", "height": 143} executionInfo={"elapsed": 604, "status": "ok", "timestamp": 1703166932634, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="hZXNAh4gtbcP" outputId="42306dd2-e055-44af-b2bf-fa890a57e34f"
print(documents[0].text)

# %% [markdown]
# To combine the text from multiple documents into a single document, we use the `Document` class from `llama_index.core`.

# %% executionInfo={"elapsed": 502, "status": "ok", "timestamp": 1703166939411, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="erEDedYUtVdn"
from llama_index.core import Document

# %% [markdown]
# Finally, we write the combined text to an HTML file.

# %%
document = Document(text="\n\n".join([doc.text for doc in documents]))

# %% executionInfo={"elapsed": 549, "status": "ok", "timestamp": 1703166946877, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="PHQ1Mmz2s0Hz"
# Write the HTML string to the file
with open('data/blog.html', "w") as file:
    file.write(document.text)

# %% [markdown] id="g238odRKmOvx"
# ## 4. Reading from Directory

# %% [markdown] id="pR8KdpIzZzhd"
# Using SimpleDirectoryReader: You can directly load all the files present in the directory or specify the multiple file names that you want to read.

# %% executionInfo={"elapsed": 368, "status": "ok", "timestamp": 1703166965264, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="4Gf6_4O3oXau"
from llama_index.core import SimpleDirectoryReader

# %% [markdown]
# We create an instance of `SimpleDirectoryReader` and specify the directory to read from.

# %% executionInfo={"elapsed": 2447, "status": "ok", "timestamp": 1703166970808, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="POJ9q5tMY9A4"
documents = SimpleDirectoryReader('./data/').load_data()

# %% [markdown]
# We can also specify individual files within the directory to read.

# %% executionInfo={"elapsed": 2503, "status": "ok", "timestamp": 1703166981837, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="BuyWTWVHZA8h"
documents = SimpleDirectoryReader(input_files=['./data/transformers.pdf',
                                               './data/transactions.csv']).load_data()

# %% [markdown]
# To access and display the text of a specific document, we reference it by its index.

# %% colab={"base_uri": "https://localhost:8080/", "height": 143} executionInfo={"elapsed": 1302, "status": "ok", "timestamp": 1703166992374, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="77J5njBAxbPO" outputId="254efac5-22ec-41c1-c390-cb5d2570d3e2"
documents[15].text

# %%

# %% [markdown]
# These examples demonstrate the flexibility of LlamaIndex in handling various data sources, making it a powerful tool for data processing and analysis.

# %% [markdown] id="LKSTRCJxanCk"
# ### You can find various data loaders [here](https://llamahub.ai/).
