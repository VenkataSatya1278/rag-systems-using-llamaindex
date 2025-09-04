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

# %% [markdown] id="D1Y3gSVdHdzb"
# # Router Query Engine
#
# Routers serve as specialized modules designed to process a user's query and select from a set of predefined "choices," characterized by their metadata.
#
# There are two primary types of core router modules:
#
# 1. **LLM Selectors:** These selectors present the available choices as a text prompt, utilizing the LLM text completion endpoint for decision-making.
#
# 2. **Pydantic Selectors:** Here, choices are passed in the form of Pydantic schemas to a function-calling endpoint. The results are then returned as Pydantic objects.

# %% [markdown] id="pV4KnOmXUPC-"
# ## Setup

# %% id="R_Ttz5WDI8M6"
# NOTE: This is ONLY necessary in jupyter notebook.
# Details: Jupyter runs an event-loop behind the scenes.
#          This results in nested event-loops when we start an event-loop to make async queries.
#          This is normally not allowed, we use nest_asyncio to allow it for convenience.
import nest_asyncio

nest_asyncio.apply()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9454, "status": "ok", "timestamp": 1698334064081, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="LGEOh3sdFuQ-" outputId="7be5e69a-c15c-4c5c-ae87-2b5a230fff0d"
import logging
import sys

# Set up the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set logger level to INFO

# Clear out any existing handlers
logger.handlers = []

# Set up the StreamHandler to output to sys.stdout (Colab's output)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)  # Set handler level to INFO

# Add the handler to the logger
logger.addHandler(handler)

from llama_index.core import VectorStoreIndex, SummaryIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from IPython.display import display, HTML

# %%
import os 
from dotenv import load_dotenv, find_dotenv

# %%
load_dotenv('/home/santhosh/Projects/courses/Pinnacle/.env')

# %%
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# %% [markdown] id="_J2AEezZP4Uf"
# ## Download Data

# %%
# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/9607a05a923ddf07deee86a56d386b42943ce381/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# %% [markdown] id="7f1Awi4MVZRf"
# ## Load data

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3248, "status": "ok", "timestamp": 1698334094489, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="TozkIS3mF8DC" outputId="27a9d1b7-47db-45d7-a67c-09101a6577cd"
# load documents
documents = SimpleDirectoryReader("data/paul_graham").load_data()

nodes = Settings.node_parser.get_nodes_from_documents(documents, chunk_size=1024)

# %% [markdown] id="e4LNrmhrK4xe"
# ## Define Summary Index and Vector Index over Same Data

# %%
embed_model = OpenAIEmbedding(model='text-embedding-3-small')

# %% id="B9C2Gm1UF8r3"
# Summary Index for summarization questions
summary_index = SummaryIndex(nodes, embed_model=embed_model)

# Vector Index for answering specific context questions
vector_index = VectorStoreIndex(nodes, embed_model=embed_model)

# %% [markdown] id="voktqsQNLApj"
# ## Define Query Engines.
#
# 1. Summary Index Query Engine.
# 2. Vector Index Query Engine.

# %% id="uLMKbMAUGA9f"
# Summary Index Query Engine
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)

# Vector Index Query Engine
vector_query_engine = vector_index.as_query_engine()

# %% [markdown] id="vzm7L0MzYQUR"
# ## Build summary index and vector index tools

# %% id="fg4aOwPaGNxb"
from llama_index.core.tools.query_engine import QueryEngineTool

# Summary Index tool
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Useful for summarization questions related to Paul Graham eassy on What I Worked On.",
)

# Vector Index tool
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context from Paul Graham essay on What I Worked On.",
)

# %% [markdown] id="F0zHAQS_LF3r"
# ## Define Router Query Engine
#
# Various selectors are at your disposal, each offering unique characteristics.
#
# Pydantic selectors, supported exclusively by gpt-4-0613 and the default gpt-3.5-turbo-0613, utilize the OpenAI Function Call API. Instead of interpreting raw JSON, they yield pydantic selection objects.
#
# On the other hand, LLM selectors employ the LLM to generate a JSON output, which is then parsed to query the relevant indexes.
#
# For both selector types, you can opt to route to either a single index or multiple indexes.

# %% [markdown] id="jTncjEj2LH88"
# ## PydanticSingleSelector
#
# Use the OpenAI Function API to generate/parse pydantic objects under the hood for the router selector.

# %% id="GGjl2y5QGRcW"
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.selectors.pydantic_selectors import PydanticMultiSelector, PydanticSingleSelector

# %%
# Create Router Query Engine
query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11130, "status": "ok", "timestamp": 1698334234628, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="0hywia-DGTx2" outputId="3810695c-327b-4b3c-fea9-def722100d8e"
response = query_engine.query("What is the summary of the document?")

# %% colab={"base_uri": "https://localhost:8080/", "height": 153} executionInfo={"elapsed": 418, "status": "ok", "timestamp": 1698334241283, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="sriu-0zoLR7q" outputId="88cab857-1e95-4eaf-d119-cee06013892d"
display(HTML(f'<p style="font-size:14px">{response.response}</p>'))

# %% [markdown] id="5zkEOzNYLUQ_"
# ## LLMSingleSelector
#
# Utilize OpenAI (or another LLM) to internally interpret the generated JSON and determine a sub-index for routing.

# %% id="S6i5dK4Muuaz"
# Create Router Query Engine
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10361, "status": "ok", "timestamp": 1698336346707, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="_y-1ztZ6utGp" outputId="5bc8a8d9-8a6d-408e-a63a-ae1bbc445b97"
response = query_engine.query("What is the summary of the document?")

# %% colab={"base_uri": "https://localhost:8080/", "height": 128} executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1698336350750, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="vwgHKW-qusYR" outputId="130038dc-a965-4dc2-e793-7e0487fedcb1"
display(HTML(f'<p style="font-size:14px">{response.response}</p>'))

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4036, "status": "ok", "timestamp": 1698336369564, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="rMgPIzfvuriI" outputId="22ea75dc-dc00-4482-af00-d972f67deeb5"
response = query_engine.query("What did Paul Graham do after RICS?")

# %% colab={"base_uri": "https://localhost:8080/", "height": 54} executionInfo={"elapsed": 403, "status": "ok", "timestamp": 1698336376616, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="Bzo6X5W2uqpe" outputId="c1e75d00-d12d-4441-e046-1a329aea2d23"
display(HTML(f'<p style="font-size:14px">{response.response}</p>'))

# %% [markdown] id="8MqCihafLZyE"
# ## PydanticMultiSelector
#
# If you anticipate queries being directed to multiple indexes, it's advisable to use a multi-selector. This selector dispatches the query to various sub-indexes and subsequently aggregates the responses through a summary index to deliver a comprehensive answer.

# %% [markdown] id="nEV_L_mkWL7V"
# ## Let's create a simplekeywordtable index and corresponding tool.

# %% id="Hexv6i0runj8"
from llama_index.core import SimpleKeywordTableIndex

keyword_index = SimpleKeywordTableIndex(nodes)

keyword_query_engine = keyword_index.as_query_engine()

keyword_tool = QueryEngineTool.from_defaults(
    query_engine=keyword_query_engine,
    description="Useful for retrieving specific context using keywords from Paul Graham essay on What I Worked On.",
)

# %% [markdown] id="FBLyUgEYWTcc"
# ## Build a router query engine.

# %% id="as9REJo7ulzu"
query_engine = RouterQueryEngine(
    selector=PydanticMultiSelector.from_defaults(),
    query_engine_tools=[
        vector_tool,
        keyword_tool,
        summary_tool
    ],
)

# %% id="HDIFO7v9uixW"
# This query could use either a keyword or vector query engine, so it will combine responses from both
response = query_engine.query(
    "What were noteable events and people from the authors time at Interleaf and YC?"
)

# %% id="EuI1xYLN-rKe"
display(HTML(f'<p style="font-size:14px">{response.response}</p>'))

# %%
