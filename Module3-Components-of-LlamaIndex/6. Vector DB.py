# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: .training-env
#     language: python
#     name: python3
# ---

# %% [markdown] id="ACfLrpwlo8Nl"
# # Vector Databases

# %% [markdown]
#
# - Flow: Documents ->chunks -> Embeddings -> Inded -> Stored in Vector database.
# - Types of Storage : Vector Stores(default), Document stores, Index stores, Graph stores, chat Stores
# - Challenges with InMemory Vector Store: 
# - Vectore Databases : Pinecone, qdrant, chroma, elasticsearch, weaviate, redis, pgvector etc.
# - Features of vector DB : DataManagement, Scalability, Real time, Backup and Security, and Metadata Storage and filtering.
# - Opensource and closed vector databases.
# - Common Vector Database search Techniques/Algorithm: Locality-Sensitive Hashing(LSH), Hierachical Navigable Small World(HNSW), Annoy(Approximate Nearest Neighbors Oh Yeah).
# - Which technique is used by different vector databases? (example:chroma uses HNSW)
# - How to find which techique to be used for a specific usecase?

# %% [markdown]
# Download the required packages by executing the below commands in either Anaconda Prompt (in Windows) or Terminal (in Linux or Mac OS)

# %% [markdown] id="VDcwqqSp8TdF"
# pip install chromadb llama-index-vector-stores-chroma pinecone-client llama-index-vector-stores-pinecone

# %%
# !pip install chromadb llama-index-vector-stores-chroma pinecone-client llama-index-vector-stores-pinecone

# %% [markdown] id="dnDtMwXk_-__"
# ## Setup

# %% id="cKlax-updNW-"
import os

# %%
from dotenv import load_dotenv, find_dotenv
load_dotenv('D:/Training/FAA-Training/Beyond-the-Prompt-Practical-RAG-for-Real-World-AI/RAG-systems-using-LlamaIndex/RAG-System-Using-LamaIndex/.env')

# %% [markdown]
# - Creat a pinecone api key

# %%
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']

# %% [markdown] id="yLtBXZ0xDtmQ"
# ## Download Data
#

# %%
# !mkdir data
# !wget "https://arxiv.org/pdf/1706.03762" -O './data/transformer.pdf'

# %% [markdown] id="faUENzMLEBfY"
# ## Load Data

# %% id="gGfPPk4gBAkQ"
from pathlib import Path
from llama_index.readers.file import PDFReader

# %%
loader = PDFReader()
documents = loader.load_data(file=Path('./data/transformers.pdf'))

# %%
len(documents)

# %% [markdown] id="DrsOfyF-5m8y"
# # VectorStoreIndex

# %% id="91b7xZYA9xjj"
from llama_index.core import VectorStoreIndex

# %% id="f30twJjzoWzy"
# Construct an index by loading documents into a VectorStoreIndex.
index = VectorStoreIndex.from_documents(documents)

# %% [markdown] id="Y4GXWBGZjhG_"
# Save index to local disk

# %%
pwd

# %% id="KNOfaLMF8Q2u"
# !mkdir index
index.storage_context.persist(persist_dir="./index")

# %%
whos

# %%
del index

# %%
whos

# %%
index

# %% id="X2WtMBtb8Tar"
from llama_index.core import StorageContext, load_index_from_storage

# %% id="jL8YwD5VjdIQ"
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./index")

# %% id="mfaY0TVkjdLG"
# load index
index = load_index_from_storage(storage_context)

# %%
whos

# %% [markdown] id="AVT3WNfw8c9L"
# # Using ChromDB

# %% id="4ebysIhS8WoE"
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# %%
# load some documents
documents = SimpleDirectoryReader(input_files=['./data/transformers.pdf']).load_data()

# %%
# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db")

# %%
# create collection
chroma_collection = db.get_or_create_collection("quickstart")

# %%
# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# %%
# create your index
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# %% [markdown] id="UQ4_QgrvkYa2"
# # Using PineCone Vector DB

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 969, "status": "ok", "timestamp": 1703156895961, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="xmqkV-3bkber" outputId="509fb8ef-e8ad-47bb-bfc1-d3d133a762a5"
from pinecone import Pinecone, PodSpec

# %%
pc = Pinecone(api_key=PINECONE_API_KEY)

# %%
pc.create_index(
  name="quickstart",
  dimension=1536,
  metric="cosine",
  spec=PodSpec(
    environment="gcp-starter"
  )
)

# %% id="goIECdTIlIkh"
pinecone_index = pc.Index("quickstart")

# %% id="WbBdMmFolInN"
# load some documents
documents = SimpleDirectoryReader(input_files=['./data/transformers.pdf']).load_data()

# %% id="Zp7D9-aElIpS"
from llama_index.vector_stores.pinecone import PineconeVectorStore
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# %% id="uZyGnt6TnZID"
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["eb068dd4db2c4dfbb60a1b2d7eb4313e", "676b8501c9cc46dca32705bf01a03678", "07a9d06acde14ad8b0571b0f3ee6d767", "b3386b1e92c04246a84483bb8b79f450", "e977a0f84b04495f9ce3f81845a359db", "3e368b49419b44f585e48ef606d14e1b", "9ed036595f4d4b25965b1e5b8a975b3a", "96369c12b4d044f6bd447a2f028f01a7", "998529df8a994ed4a85ea5d8a87c26a8", "2130220d2a5d4c19aa064110b24b1e6c", "d3271e5fb19c49b5a842a3e00f4d9618"]} executionInfo={"elapsed": 21547, "status": "ok", "timestamp": 1703156976385, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="An0RcriHnZK0" outputId="c6a0c067-c67c-45bd-c497-190960a30f48"
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# %% [markdown] id="tb9AScYO8yoV"
# https://docs.llamaindex.ai/en/stable/understanding/storing/storing.html
