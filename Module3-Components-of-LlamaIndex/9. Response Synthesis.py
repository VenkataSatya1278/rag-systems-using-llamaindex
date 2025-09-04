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

# %% [markdown] id="PPkG_5ist0qS"
# # Response Synthesis
#
# In this notebook, we will explore the response synthesis module, focusing on generating responses using various modes and indices.

# %% [markdown]
# ## Content
# - Function of Response Synthesizer.
# - Different types of Response synthesizer: Refine, Compact, Accumulate, Compact Accumulater, Tree Summarize 
# - How each type works? What are challenges with each one?
# - 

# %% [markdown] id="dnDtMwXk_-__"
# ## Setup

# %% id="cKlax-updNW-"
import os

# %%
from dotenv import load_dotenv, find_dotenv
load_dotenv('D:/Training/FAA-Training/Beyond-the-Prompt-Practical-RAG-for-Real-World-AI/RAG-systems-using-LlamaIndex/RAG-System-Using-LamaIndex/.env')

# %%
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# %% [markdown] id="yLtBXZ0xDtmQ"
# ## Download Data

# %%
# !mkdir data
# !powershell -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/1706.03762.pdf' -OutFile './data/transformers.pdf'"

# %%
from pathlib import Path
from llama_index.readers.file import PDFReader

# %% executionInfo={"elapsed": 498, "status": "ok", "timestamp": 1703166832879, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="dBLPDbZ5u5_D"
loader = PDFReader()

# %% executionInfo={"elapsed": 4233, "status": "ok", "timestamp": 1703166842594, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="wJAWQF2amw01"
documents = loader.load_data(file=Path('./data/transformers.pdf'))

# %%
len(documents)

# %% executionInfo={"elapsed": 1843, "status": "ok", "timestamp": 1703146380060, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="SkWY1KWgAqrw"
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)

# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1703146380061, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="PZLJMTP3As60"
# configure retriever
retriever = index.as_retriever()

# %% [markdown] id="iDG6HwUp5f2T"
# # Different types of response synthesizer

# %% [markdown] id="PfpT7LTnvbdD"
# ## Refine

# %% id="aL5Fsn4LsGkb"
from llama_index.core import get_response_synthesizer

# %% id="fUqf-ksCA5vK"
# configure response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="refine")

# %% [markdown] id="wYI2YJuo4HBR"
# ## Compact

# %% id="KuWXWiv-4LGm"
# configure response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="compact")

# %% [markdown] id="6dYQbgO7u6Bg"
# ## Tree Summarize

# %% id="4Bn026hFIdH4"
# configure response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

# %% [markdown] id="d8nPuYXgvJZC"
# ## Accumulate

# %% id="Iu3zlj3rN0eL"
# configure response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="accumulate")

# %% [markdown] id="sWggluo1410F"
# ## Compact Accumulate

# %% id="AbaAIQmJ40yA"
# configure response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="compact_accumulate")

# %% [markdown]
# # Next: Setting up the Query Engine
