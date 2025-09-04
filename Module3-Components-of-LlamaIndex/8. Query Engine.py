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

# %% [markdown]
# # Query Engine and Chat Engine

# %% [markdown]
# ### Content
# - Flow : user input -> Query Engine -> Output
# - Query Engine Functionality: 
# - Pros and Cons of Query Engine
# - Type of engines: Query Engine and Chat Engine. What is the differnce between them.
# - Understand when to use which Engine?

# %% [markdown] id="PPkG_5ist0qS"
# ### Setup

# %% id="cKlax-updNW-"
import os

# %%
from dotenv import load_dotenv, find_dotenv
load_dotenv('D:/Training/FAA-Training/Beyond-the-Prompt-Practical-RAG-for-Real-World-AI/RAG-systems-using-LlamaIndex/RAG-System-Using-LamaIndex/.env')

# %%
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# %% [markdown] id="yLtBXZ0xDtmQ"
# ### Download Data

# %%
# !mkdir data
# !wget "https://arxiv.org/pdf/1706.03762" -O 'data/transformers.pdf'

# %%
from pathlib import Path
from llama_index.readers.file import PDFReader

# %% executionInfo={"elapsed": 498, "status": "ok", "timestamp": 1703166832879, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="dBLPDbZ5u5_D"
loader = PDFReader()

# %% executionInfo={"elapsed": 4233, "status": "ok", "timestamp": 1703166842594, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="wJAWQF2amw01"
documents = loader.load_data(file=Path('./data/transformers.pdf'))

# %% executionInfo={"elapsed": 1843, "status": "ok", "timestamp": 1703146380060, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="SkWY1KWgAqrw"
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)

# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1703146380061, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="PZLJMTP3As60"
# configure retriever
retriever = index.as_retriever()

# %% executionInfo={"elapsed": 410, "status": "ok", "timestamp": 1703145606826, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="aL5Fsn4LsGkb"
#configure response synthesizer
from llama_index.core import get_response_synthesizer

response_synthesizer = get_response_synthesizer(response_mode="compact")

# %% [markdown] id="oYgniLGFhb7w"
# # Query Engine

# %% executionInfo={"elapsed": 408, "status": "ok", "timestamp": 1703145616963, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="dj8gJOpbhefH"
query_engine = index.as_query_engine(response_synthesizer=response_synthesizer)

# %% executionInfo={"elapsed": 2895, "status": "ok", "timestamp": 1703145627439, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="SNEkDJ5Wheht"
response = query_engine.query("Give me the authors of transformers paper")
print(response)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 386, "status": "ok", "timestamp": 1703145629947, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="a7jS1LsPhekV" outputId="b2d67472-5858-4e39-f1d5-035c1dc46d5f"
response.source_nodes

# %%
response.source_nodes[0].dict()

# %%
response.source_nodes[0].text

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1703145690294, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="vjLMF8CdZBBJ" outputId="aa05a581-9ae0-433f-c64a-43babe44bd9e"
print(response.source_nodes[0].get_content())

# %% executionInfo={"elapsed": 2404, "status": "ok", "timestamp": 1703145708369, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="jghxOHdtiGIm"
response = query_engine.query("What is the use of positional encoding?")
print(response)

# %%
response = query_engine.query("What is the use of positional encoding? Answer in approx 250 characters.")
print(response)

# %%
print(response.get_formatted_sources())

# %%
response.metadata

# %%
len(response.response)

# %% [markdown] id="8WsBJGwriqXb"
# # Chat Engine

# %% executionInfo={"elapsed": 464, "status": "ok", "timestamp": 1703145786655, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="ygBw-QSPipmc"
chat_engine = index.as_chat_engine(response_synthesizer=response_synthesizer)

# %% executionInfo={"elapsed": 7890, "status": "ok", "timestamp": 1703145800074, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="5jZHaXe_i2Nb"
response = chat_engine.chat("Give me the authors of transformers")
print(response)

# %% executionInfo={"elapsed": 1630, "status": "ok", "timestamp": 1703145810091, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="yo1jY_uyjBPY"
response = chat_engine.chat("What is the use of positional encoding?")
print(response)
