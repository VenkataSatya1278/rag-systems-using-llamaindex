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

# %% [markdown] id="307804a3-c02b-4a57-ac0d-172c30ddc851"
# # Auto-Retrieval from a Vector Database
#
# This guide shows how to perform **auto-retrieval** in LlamaIndex.
#
# Many popular vector dbs support a set of metadata filters in addition to a query string for semantic search. Given a natural language query, we first use the LLM to infer a set of metadata filters as well as the right query string to pass to the vector db (either can also be blank). This overall query bundle is then executed against the vector db.
#
# This allows for more dynamic, expressive forms of retrieval beyond top-k semantic search. The relevant context for a given query may only require filtering on a metadata tag, or require a joint combination of filtering + semantic search within the filtered set, or just raw semantic search.
#
# We demonstrate an example with Chroma, but auto-retrieval is also implemented with many other vector dbs (e.g. Pinecone, Weaviate, and more).

# %% [markdown] id="f7010b1d-d1bb-4f08-9309-a328bb4ea396"
# ## Setup
#
# We first define imports and define an empty Chroma collection.

# %% [markdown] id="31faecfb"
# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# %% executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1702812990588, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="d48af8e1"
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# %%
import os 
from dotenv import load_dotenv, find_dotenv

# %%
load_dotenv('/home/santhosh/Projects/courses/Pinnacle/.env')

# %%
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1702812990588, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="bf49ac18"
# set up OpenAI
import os
import getpass

# os.environ["OPENAI_API_KEY"] = 'YOUR OPENAI API KEY'
import openai

openai.api_key = OPENAI_API_KEY

# %% id="0ce3143d-198c-4dd2-8e5a-c5cdf94f017a"
import chromadb

# %% id="667f3cb3-ce18-48d5-b9aa-bfc1a1f0f0f6"
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.get_or_create_collection("quickstart")

# %% [markdown] id="41aa106b-8261-4a01-97c6-1b037dffa1b4"
# ## Defining Some Sample Data
#
# We insert some sample nodes containing text chunks into the vector database. Note that each `TextNode` not only contains the text, but also metadata e.g. `category` and `country`. These metadata fields will get converted/stored as such in the underlying vector db.

# %% executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1702813031268, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="0a2bcc07"
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1702813032432, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="68cbd239-880e-41a3-98d8-dbb3fab55431"
from llama_index.core.schema import TextNode

nodes = [
    TextNode(
        text=(
            "Michael Jordan is a retired professional basketball player,"
            " widely regarded as one of the greatest basketball players of all"
            " time."
        ),
        metadata={
            "category": "Sports",
            "country": "United States",
        },
    ),
    TextNode(
        text=(
            "Angelina Jolie is an American actress, filmmaker, and"
            " humanitarian. She has received numerous awards for her acting"
            " and is known for her philanthropic work."
        ),
        metadata={
            "category": "Entertainment",
            "country": "United States",
        },
    ),
    TextNode(
        text=(
            "Elon Musk is a business magnate, industrial designer, and"
            " engineer. He is the founder, CEO, and lead designer of SpaceX,"
            " Tesla, Inc., Neuralink, and The Boring Company."
        ),
        metadata={
            "category": "Business",
            "country": "United States",
        },
    ),
    TextNode(
        text=(
            "Rihanna is a Barbadian singer, actress, and businesswoman. She"
            " has achieved significant success in the music industry and is"
            " known for her versatile musical style."
        ),
        metadata={
            "category": "Music",
            "country": "Barbados",
        },
    ),
    TextNode(
        text=(
            "Cristiano Ronaldo is a Portuguese professional footballer who is"
            " considered one of the greatest football players of all time. He"
            " has won numerous awards and set multiple records during his"
            " career."
        ),
        metadata={
            "category": "Sports",
            "country": "Portugal",
        },
    ),
]

# %% [markdown] id="e8bd70be-57c7-49e2-990b-ad9a876710fb"
# ## Build Vector Index with Chroma Vector Store
#
# Here we load the data into the vector store. As mentioned above, both the text and metadata for each node will get converted into corresopnding representations in Chroma. We can now run semantic queries and also metadata filtering on this data from Chroma.

# %% executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1702813041164, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="ba1558b3"
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# %% executionInfo={"elapsed": 617, "status": "ok", "timestamp": 1702813044990, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="35369eda"
index = VectorStoreIndex(nodes, storage_context=storage_context)

# %% [markdown] id="c793dc45-5087-4dcb-b0d3-85b8e718539f"
# ## Define `VectorIndexAutoRetriever`
#
# We define our core `VectorIndexAutoRetriever` module. The module takes in `VectorStoreInfo`,
# which contains a structured description of the vector store collection and the metadata filters it supports.
# This information will then be used in the auto-retrieval prompt where the LLM infers metadata filters.

# %%
from llama_index.core.indices.vector_store.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo

# %% executionInfo={"elapsed": 361, "status": "ok", "timestamp": 1702813048316, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="bedbb693-725f-478f-be26-fa7180ea38b2"
vector_store_info = VectorStoreInfo(
    content_info="brief biography of celebrities",
    metadata_info=[
        MetadataInfo(
            name="category",
            type="str",
            description=(
                "Category of the celebrity, one of [Sports, Entertainment,"
                " Business, Music]"
            ),
        ),
        MetadataInfo(
            name="country",
            type="str",
            description=(
                "Country of the celebrity, one of [United States, Barbados,"
                " Portugal]"
            ),
        ),
    ],
)
retriever = VectorIndexAutoRetriever(
    index, vector_store_info=vector_store_info
)

# %% [markdown] id="32808a60-7bab-4e9e-944c-cfe2ed0b0e2e"
# ## Running over some sample data
#
# We try running over some sample data. Note how metadata filters are inferred - this helps with more precise retrieval!

# %% id="eeb18e9c"
retriever.retrieve("Tell me about two celebrities from United States")

# %% id="51f00cde"
retriever.retrieve("Tell me about Sports celebrities from United States")

# %% id="51f00cde"
retriever.retrieve("Tell me about Sports celebrities from United States")
