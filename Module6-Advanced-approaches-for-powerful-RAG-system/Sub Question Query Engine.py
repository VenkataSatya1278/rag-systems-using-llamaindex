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

# %% [markdown] id="NzWzjWew9zW-"
# # Compare Documents
# # SubQuestionQueryEngine

# %% id="uQcsPrXS9-qn"
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# %%
import os 
from dotenv import load_dotenv, find_dotenv

# %%
load_dotenv('/home/santhosh/Projects/courses/Pinnacle/.env')

# %%
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# %% id="Mu7ZI3_O-CDC"
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import set_global_service_context
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

# %% [markdown] id="dj4J1NYO__6L"
# # Load uber and lyft documents

# %% id="-gqVPUhF-MsC"
lyft_docs = SimpleDirectoryReader(input_files=["lyft_2021.pdf"]).load_data()
uber_docs = SimpleDirectoryReader(input_files=["uber_2021.pdf"]).load_data()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 483, "status": "ok", "timestamp": 1693039041443, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="6WRclpxW-QOX" outputId="dcbc71f8-8f6d-4f57-c56e-82efdca5ad15"
print(f'Loaded lyft 10-K with {len(lyft_docs)} pages')
print(f'Loaded Uber 10-K with {len(uber_docs)} pages')

# %% [markdown] id="c9l5XFXOACgA"
# # Build indices

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 28207, "status": "ok", "timestamp": 1693039071323, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="xBzg6G9H-QuH" outputId="0a0032e1-4682-4eaf-d619-d3b9518c4248"
lyft_index = VectorStoreIndex.from_documents(lyft_docs)
uber_index = VectorStoreIndex.from_documents(uber_docs)

# %% [markdown] id="OG9N9PT-AG_8"
# # Basic QA

# %% id="tMTcHwSs-Uf1"
lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)


# %% id="Og6bs-Nm-U61"
uber_engine = uber_index.as_query_engine(similarity_top_k=3)


# %% id="ImwBL7W2-XyR"
response = await lyft_engine.aquery('What is the revenue of Lyft in 2021? Answer in millions with page reference')


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1693039117968, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="v-fLA2rG_lM-" outputId="13e2917a-c84a-40c6-b791-5671a8c00986"
print(response)

# %% id="WUW9z_MN-YUt"
response = await uber_engine.aquery('What is the revenue of Uber in 2021? Answer in millions, with page reference')


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1693039119557, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="LSyKvMZd_mkw" outputId="600f3ec1-6982-4a2d-81d5-a5d7de11b69d"
print(response)

# %% [markdown] id="AN6j0tLYAJya"
# # For comparing between uber and lyft

# %% id="7Q6NHot0-dGY"
query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(name='lyft_10k', description='Provides information about Lyft financials for year 2021')
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(name='uber_10k', description='Provides information about Uber financials for year 2021')
    ),
]

s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10668, "status": "ok", "timestamp": 1693039133576, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="TBJbICbI-f9e" outputId="e1ee6101-a1fc-47a9-f7c9-c47d1a206786"
response = await s_engine.aquery('Compare and contrast the customer segments and geographies that grew the fastest')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1693039133578, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="Qn50QRnT_oWw" outputId="894c1986-51a3-4b37-943b-636ec4851ff7"
print(response)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6285, "status": "ok", "timestamp": 1693039139858, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="liOSDYiK-iy6" outputId="f3797b93-5388-4be9-dd24-87a0d9d57087"
response = await s_engine.aquery('Compare revenue growth of Uber and Lyft from 2020 to 2021')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1693039139859, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="trxesUfp_kCf" outputId="ab3e531f-8ba0-48ea-9534-8c206b38c049"
print(response)
