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

# %% [markdown] id="GS4fjNC0WSmz"
# # Reciprocal Rerank Fusion Retriever
#
# In this example, we walk through how you can combine retireval results from multiple queries and multiple indexes.
#
# The retrieved nodes will be reranked according to the `Reciprocal Rerank Fusion` algorithm demonstrated in this [paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). It provides an effecient method for rerranking retrieval results without excessive computation or reliance on external models.

# %% [markdown]
# pip install llama-index-retrievers-bm25

# %%
import os 
from dotenv import load_dotenv, find_dotenv

# %%
load_dotenv('/home/santhosh/Projects/courses/Pinnacle/.env')

# %%
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# %% [markdown] id="L07wbltxWSm0"
# ## Setup

# %% [markdown] id="6K56TOrwWSm1"
# Download Data

# %%
# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/9607a05a923ddf07deee86a56d386b42943ce381/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# %% id="dhxBg_QNWSm1"
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# %% [markdown] id="gRT-2OKrWSm1"
# Next, we will setup a vector index over the documentation.

# %% id="WLYeL3XpWSm1"
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents, chunk_size=256)

# %% [markdown] id="4MB6MdSvWSm2"
# ## Create a Hybrid Fusion Retriever
#
# In this step, we fuse our index with a BM25 based retriever. This will enable us to capture both semantic relations and keywords in our input queries.
#
# Since both of these retrievers calculate a score, we can use the reciprocal rerank algorithm to re-sort our nodes without using an additional models or excessive computation.
#
# This setup will also query 4 times, once with your original query, and generate 3 more queries.
#
# By default, it uses the following prompt to generate extra queries:
#
# ```python
# QUERY_GEN_PROMPT = (
#     "You are a helpful assistant that generates multiple search queries based on a "
#     "single input query. Generate {num_queries} search queries, one on each line, "
#     "related to the following input query:\n"
#     "Query: {query}\n"
#     "Queries:\n"
# )
# ```

# %% [markdown] id="EnucEpMWWSm2"
# First, we create our retrievers. Each will retrieve the top-2 most similar nodes:

# %%
from llama_index.retrievers.bm25 import BM25Retriever

# %% id="8mJUQATqWSm2"
vector_retriever = index.as_retriever(similarity_top_k=2)

bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore, similarity_top_k=2
)

# %% [markdown] id="uo_0Le5mWSm2"
# Next, we can create our fusion retriever, which well return the top-2 most similar nodes from the 4 returned nodes from the retrievers:

# %% id="R9fJ2c_7WSm2"
from llama_index.core.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=2,
    num_queries=4,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    use_async=True,
    verbose=True,
    # query_gen_prompt="...",  # we could override the query generation prompt here
)

# %% id="7K_E_Gl4WSm2"
# apply nested async to run in a notebook
import nest_asyncio

nest_asyncio.apply()

# %% id="eJgXMQKJWSm2" outputId="04d20c4c-fb9d-42c0-c827-5ad921a1402f"
nodes_with_scores = retriever.retrieve(
    "What happened at Interleafe and Viaweb?"
)

# %% id="c19C3uTmWSm2" outputId="4c113ec3-a59e-4b19-92be-cae8bc589a45"
for node in nodes_with_scores:
    print(f"Score: {node.score:.2f} - {node.text}...\n-----\n")

# %% [markdown] id="SxZ4tCZ9WSm2"
# As we can see, both retruned nodes correctly mention Viaweb and Interleaf!

# %% [markdown] id="FRiLRM9KWSm2"
# ## Use in a Query Engine!
#
# Now, we can plug our retriever into a query engine to synthesize natural language responses.

# %% id="Fn4ySBaAWSm2"
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(retriever)

# %% id="AvXwWymCWSm3" outputId="bfd6c6ae-5805-425f-d082-e787fadcb034"
response = query_engine.query("What happened at Interleafe and Viaweb?")

# %% id="YQ5PZNiAWSm3" outputId="d11d79ba-0bde-45b9-fea2-ea80a91d7d9c"
from llama_index.core.response.notebook_utils import display_response

display_response(response)

# %%
