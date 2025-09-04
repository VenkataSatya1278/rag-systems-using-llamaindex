# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="roVWl5PaKL0d"
# # Ingestion Pipeline And Token Counting.
#
# In this notebook, we will explore the concept of Ingestion Pipeline in LlamaIndex which helps you to easily create nodes, subsequently index and query them. Additionally we will even see how you can estimate the cost of your pipeline using number of tokens used.
#
# 1. Ingestion Pipeline - Easily Ingesting data.
# 2. Transformation caching Inmemory.
# 3. Custom Transformations.
# 4. Tokenization and Token Counting.
#
# We will delve into each of these features in detail throughout the notebook.

# %% [markdown] id="381jj00E7_gD"
# ## Setup

# %% id="40d399c4-c93c-41bf-9a47-48aefabb75e3"
import nest_asyncio
nest_asyncio.apply()

# %%
import os
# from dotenv import load_dotenv, find_dotenv
# load_dotenv('D:/.env')
# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# %%

# %% [markdown] id="TJ_66T7F9WkN"
# ## Download Data
#
# We will use Paul Graham essay text for this tutorial.

# %% executionInfo={"elapsed": 500, "status": "ok", "timestamp": 1703423696702, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="7u6F6LEWLWtP"
# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/9607a05a923ddf07deee86a56d386b42943ce381/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# %% [markdown] id="mDnFT1Za_Wgo"
# ## Load Data

# %% executionInfo={"elapsed": 8197, "status": "ok", "timestamp": 1703443654604, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="GEsiTR-j_YCz"
from llama_index.core import SimpleDirectoryReader

documents= SimpleDirectoryReader('./data/paul_graham/').load_data()

# %% [markdown] id="9bIyj5uI-00U"
# ## Ingestion Pipeline - Easily Ingesting data.
#
# An `IngestionPipeline` uses a new concept of `Transformations` that are applied to input data.
#
# The `Transformations` could be any of the following:
#
# 1. text splitter
# 2. node parser
# 3. metadata extractor
# 4. embeddings model
#
# Once the data is ingested you can build index and start querying.

# %% executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1703443689420, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="bMgNYMS2-s4J"
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline

# %% executionInfo={"elapsed": 1114, "status": "ok", "timestamp": 1703443704673, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="rIiJKUyfBALP"
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=100),
    ]
)
nodes = pipeline.run(documents=documents)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 385, "status": "ok", "timestamp": 1703443707232, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="JRM9z7FBJA8q" outputId="c47ffbeb-65f6-49ab-e299-fa25733e0753"
nodes[0]

# %%
nodes[0].metadata

# %%
len(documents)

# %%
len(nodes)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1999, "status": "ok", "timestamp": 1703443745776, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="AeFADgmEJM_8" outputId="4c640ae0-f14c-4013-dddc-8d54e1581a46"
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=100),
        TitleExtractor()
    ]
)
nodes = pipeline.run(documents=documents)

# %%
nodes[0].metadata

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 391, "status": "ok", "timestamp": 1703443748636, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="JIaXOif7JA8q" outputId="962644c0-f3fa-46fc-f458-0d618ab14d84"
nodes[0].metadata['document_title']

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 382, "status": "ok", "timestamp": 1703443753385, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="Ciz-eDg-JA8q" outputId="9f0231bb-4f06-47f4-a7e1-c3a4fa252622"
len(nodes)

# %%
nodes[0].embedding

# %% [markdown] id="2z7lFfLEJA8r"
# Let's include embeddings.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2830, "status": "ok", "timestamp": 1703443764800, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="aaOmXotaFe0i" outputId="e0538a34-ee9b-4df3-c0f7-1cba2c159512"
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=100),
        TitleExtractor(),
        OpenAIEmbedding(model='text-embedding-3-small')
    ]
)
nodes = pipeline.run(documents=documents)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1019, "status": "ok", "timestamp": 1703443767755, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="lQeiKYFLJA8s" outputId="8cc8446c-a580-4818-8dd8-1882f8e316d3"
nodes[0].embedding[:10]

# %%

# %% [markdown] id="r3IFurypJA8t"
# [Async Ingestion Pipeline + Metadata Extraction.](https://docs.llamaindex.ai/en/latest/examples/ingestion/async_ingestion_pipeline.html#)

# %% [markdown] id="e6Dj421JGAMM"
# ## Transformation Caching
#
# Every time you execute the same IngestionPipeline object, it stores a cache of the hash combining input nodes and transformations along with their respective output. In future runs, if the cache matches, the pipeline skips the transformation and uses the cached result. This accelerates repeated executions and aids in quicker iteration when selecting transformations.

# %% executionInfo={"elapsed": 371, "status": "ok", "timestamp": 1703443886206, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="23zbMzqGD1x7"
from llama_index.core.ingestion import IngestionCache

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2649, "status": "ok", "timestamp": 1703443894554, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="C__wRox-GIDs" outputId="829d2ae1-5332-44f1-9670-7a34f9e6e21a"
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=100),
        TitleExtractor(),
    ]
)
nodes = pipeline.run(documents=documents)

# %% executionInfo={"elapsed": 364, "status": "ok", "timestamp": 1703443897198, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="7ivtr-qyGPVQ"
# save and load
pipeline.cache.persist("./llama_cache.json")
new_cache = IngestionCache.from_persist_path("./llama_cache.json")

# %% executionInfo={"elapsed": 396, "status": "ok", "timestamp": 1703443903437, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="wTONJNxjGVHa"
new_pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=100),
        TitleExtractor(),
    ],
    cache=new_cache,
)

# %% [markdown] id="A--06eFbGgME"
# ### Now it will run instantly due to the cache.
#
# Will be very useful when extracting metadata and also creating embeddings

# %% executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1703443910388, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="qnba82GCGYF2"
nodes = new_pipeline.run(documents=documents)

# %% [markdown] id="UM7UGcjuGx9O"
# Now let's add embeddings to it. You will observe that the parsing of nodes, title extraction is loaded from cache and OpenAI embeddings are created now.

# %% executionInfo={"elapsed": 1379, "status": "ok", "timestamp": 1703443944878, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="6HynWDfcGoaJ"
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=100),
        TitleExtractor(),
        OpenAIEmbedding()
    ],
    cache=new_cache,
)
nodes = pipeline.run(documents=documents)

# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1703443957111, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="qGHeGxQ3G2pR"
# save and load
pipeline.cache.persist("./nodes_embedding.json")
nodes_embedding_cache = IngestionCache.from_persist_path("./nodes_embedding.json")

# %% executionInfo={"elapsed": 626, "status": "ok", "timestamp": 1703443972759, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="1KORw9rdHiFC"
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=100),
        TitleExtractor(),
        OpenAIEmbedding(model='text-embedding-3-small')
    ],
    cache=nodes_embedding_cache,
)

# Will load it from the cache as the transformations are same.
nodes = pipeline.run(documents=documents)

# %%

# %% [markdown] id="c5fRtRkTMyIY"
# ## Custom Transformations
#
# Implementing custom transformations is pretty easy.
#
# Let's include a transformation that removes special characters from the text before generating embeddings.
#
# The primary requirement for transformations is that they should take a list of nodes as input and return a modified list of nodes.

# %% executionInfo={"elapsed": 441, "status": "ok", "timestamp": 1703444057038, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="pKfoi8MXMz6A"
from llama_index.core.schema import TransformComponent
import re

class TextCleaner(TransformComponent):
  def __call__(self, nodes, **kwargs):
    for node in nodes:
      node.text = re.sub(r'[^0-9A-Za-z ]', "", node.text)
    return nodes

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=100),
        TextCleaner(),
    ],
)

nodes = pipeline.run(documents=documents)

# %%

# %%

# %% [markdown] id="aun4jxALO2jz"
# ## Tokenization and Token Counting

# %% executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1703444163495, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="8Lyxoiy2OiBl"
import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings

# %%
token_counter = TokenCountingHandler(tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode)

# If you plan to use different LLM model you can set in following way
# from transformers import AutoTokenizer
# tokenizer=AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").encode

# %%
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.2)

# %%
# Callback manager handles callbacks for events within LlamaIndex.
Settings.callback_manager = CallbackManager([token_counter])

# %%
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex(nodes)

# %%
token_counter.total_embedding_token_count

# %%
# reset the counts at your discretion!
token_counter.reset_counts()
print(token_counter.total_embedding_token_count)

# %%
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

# %%
print(response.response)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2356, "status": "ok", "timestamp": 1703444200531, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="HCZk2qU7QwIG" outputId="521fab13-ba36-4f21-dd44-02d31e794b9f"
print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
)

# %%
