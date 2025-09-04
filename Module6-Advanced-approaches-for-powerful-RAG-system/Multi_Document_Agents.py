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

# %% [markdown] id="43497beb-817d-4366-9156-f4d7f0d44942"
# # Multi Document Agents
#
# This guide shows building Multi Document Agents by combining recursive retrieval and "document agents".
#
# There are two motivating factors that lead to solutions for better retrieval:
# - Decoupling retrieval embeddings from chunk-based synthesis. Oftentimes fetching documents by their summaries will return more relevant context to queries rather than raw chunks. This is something that recursive retrieval directly allows.
# - Within a document, users may need to dynamically perform tasks beyond fact-based question-answering. We introduce the concept of "document agents" - agents that have access to both vector search and summary tools for a given document.

# %% [markdown] id="9be00aba-b6c5-4940-9825-81c5d2cd2f0b"
# ### Setup and Download Data
#
# In this section, we'll define imports and then download Wikipedia articles about different cities. Each article is stored separately.

# %% [markdown]
# pip install llama-index-agent-openai

# %%
import os 
from dotenv import load_dotenv

# %%
load_dotenv('/home/santhosh/Projects/courses/Pinnacle/.env')

# %%
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# %% id="e41e9905-77a9-44c5-88ac-c7a4d08a4612"
from llama_index.core import VectorStoreIndex, SummaryIndex, SimpleKeywordTableIndex, SimpleDirectoryReader
from llama_index.core.schema import IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI

# %% id="e4343cf7-eec9-4a67-b5be-c72dbe3280a7"
wiki_titles = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai"]

# %% id="6d261882-6793-4eca-ad93-d94d2061e388"
from pathlib import Path

import requests

for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            # 'exintro': True,
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    data_path = Path("data")
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

# %% id="5bf0c13b-0d77-43e8-8c1c-84258299a494"
# Load all wiki documents
city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()

# %% [markdown] id="6189aaf4-2eb7-40bc-9e83-79ce4f221b4b"
# ## Define LLM + Callback Manager

# %% id="dd6e5e48-91b9-4701-a85d-d98c92323350"
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")

# %% [markdown] id="976cd798-2e8d-474c-922a-51b12c5c6f36"
# ## Build Document Agent for each Document
#
# In this section we define "document agents" for each document.
#
# First we define both a vector index (for semantic search) and summary index (for summarization) for each document. The two query engines are then converted into tools that are passed to an OpenAI function calling agent.
#
# This document agent can dynamically choose to perform semantic search or summarization within a given document.
#
# We create a separate document agent for each city.

# %%
from llama_index.agent.openai import OpenAIAgent

# %% id="eacdf3a7-cfe3-4c2b-9037-b28a065ed148"
# Build agents dictionary
agents = {}

for wiki_title in wiki_titles:
    # build vector index
    vector_index = VectorStoreIndex.from_documents(
        city_docs[wiki_title], llm=llm
    )
    # build summary index
    summary_index = SummaryIndex.from_documents(
        city_docs[wiki_title], llm=llm
    )
    # define query engines
    vector_query_engine = vector_index.as_query_engine()
    list_query_engine = summary_index.as_query_engine()

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=(
                    "Useful for summarization questions related to"
                    f" {wiki_title}"
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=list_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=(
                    f"Useful for retrieving specific context from {wiki_title}"
                ),
            ),
        ),
    ]

    # build agent
    function_llm = OpenAI(model="gpt-3.5-turbo")
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
    )

    agents[wiki_title] = agent

# %% [markdown] id="899ca55b-0c02-429b-a765-8e4f806d503f"
# ## Build Recursive Retriever over these Agents
#
# Now we define a set of summary nodes, where each node links to the corresponding Wikipedia city article. We then define a `RecursiveRetriever` on top of these Nodes to route queries down to a given node, which will in turn route it to the relevant document agent.
#
# We finally define a full query engine combining `RecursiveRetriever` into a `RetrieverQueryEngine`.

# %% id="6884ff15-bf40-4bdd-a1e3-58cbd056a12a"
# define top-level nodes
nodes = []
for wiki_title in wiki_titles:
    # define index node that links to these agents
    wiki_summary = (
        f"This content contains Wikipedia articles about {wiki_title}. Use"
        " this index if you need to lookup specific facts about"
        f" {wiki_title}.\nDo not use this index if you want to analyze"
        " multiple cities."
    )
    node = IndexNode(text=wiki_summary, index_id=wiki_title)
    nodes.append(node)

# %% id="3eabd221-3c24-468b-9dbe-19689faf57fc"
# define top-level retriever
vector_index = VectorStoreIndex(nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# %% id="f820fc10-66df-4cbe-b907-71d7fff73a71"
# define recursive retriever
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

# %% id="b36124b4-cf58-4fc6-bccd-647a73f72af1"
# note: can pass `agents` dict as `query_engine_dict` since every agent can be used as a query engine
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=agents,
    verbose=True,
)

# %% [markdown] id="85dc98fe-45ff-463b-9a70-8146a9051b28"
# #### Define Full Query Engine
#
# This query engine uses the recursive retriever + response synthesis module to synthesize a response.

# %% id="9b95e8a8-4fa6-4977-a8f4-ea100e3107ed"
response_synthesizer = get_response_synthesizer(
    response_mode="compact",
)
query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever,
    response_synthesizer=response_synthesizer,
    llm=llm,
)

# %% [markdown] id="8dedb927-a992-4f21-a0fb-4ce4361adcb3"
# ## Running Example Queries

# %% colab={"base_uri": "https://localhost:8080/"} id="8e743c62-7dd8-4ac9-85a5-f1cbc112a79c" outputId="405842ad-0b4d-4f44-8c5f-ac0ef309aa04"
# should use Delhi agent -> Summary tool
response = query_engine.query("Summarize history of Delhi")

# %% colab={"base_uri": "https://localhost:8080/"} id="a4ce2a76-5779-4acf-9337-69109dae7fd6" outputId="aed73eb0-3c7c-4919-e2f0-691738fa1351"
print(response)

# %% colab={"base_uri": "https://localhost:8080/"} id="aa3d98ab-cb82-4473-ab2b-bc8a17e1b86a" outputId="e53f469b-e1a9-4647-f806-8fb1c7c5f53e"
# should use Hyderabad agent -> vector tool
response = query_engine.query("Who is current Mayor of Hyderabad city?")

# %% colab={"base_uri": "https://localhost:8080/"} id="d476c54b-98af-4d2a-8f17-4baa37d0d360" outputId="19688f58-fbf7-4f20-c925-94afa4dce528"
print(response)

# %% colab={"base_uri": "https://localhost:8080/"} id="ee6ef20c-3ccc-46c3-ad87-667138d78d5d" outputId="bf73d758-ba09-4c2e-b82e-a97b6805570d"
# should use Chennai agent -> summary tool
response = query_engine.query(
    "Give me a summary on all the positive aspects of Chennai"
)

# %% colab={"base_uri": "https://localhost:8080/"} id="cfe1dd4c-8bfd-43d0-99bc-ca60861dc418" outputId="60e4323e-0d00-497e-8918-74303bbeaadc"
print(response)

# %% colab={"base_uri": "https://localhost:8080/"} id="lqSURoTN8AO3" outputId="9cef2bb4-1536-49c5-f4cb-65b3fea86962"
# should use Mumbai agent -> summary tool
response = query_engine.query("Summarize about British rule in Mumbai history")

# %% colab={"base_uri": "https://localhost:8080/"} id="7NwqKQiM81o3" outputId="731d5727-504e-4682-9a4f-62894d20e38f"
print(response)

# %% id="8zn5rcl7n1Ca"
