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
# # Indexing & Retrieval

# %% [markdown]
# ## Content
# - What is Index?
# - Why do we need Index in a RAG system? Efficiency, Scalability, Accurarcy, Real-time Performance
# - Index Types -> Summary Index, Vector Store Index, Document Summary Index, Tree Index, Keywork Table Index.
# - What is the usage of each Index type?
# - Retrieval modes for differnet Indexes -> Summary Index, Document summary Index, Tree Index, Keywork Table Index.
# - Can we go with Hybrid Indexing>
# - 
#

# %%
# !pip install dotenv
# !pip install llama-index-embeddings-gemini llama-index-embeddings-cohere 
# !pip install llama_index.embeddings.openai

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
# !wget "https://arxiv.org/pdf/1706.03762" -O 'data/transformers.pdf'

# %%
from pathlib import Path
from llama_index.readers.file import PDFReader

# %% executionInfo={"elapsed": 498, "status": "ok", "timestamp": 1703166832879, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="dBLPDbZ5u5_D"
loader = PDFReader()

# %% executionInfo={"elapsed": 4233, "status": "ok", "timestamp": 1703166842594, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="wJAWQF2amw01"
documents = loader.load_data(file=Path('./data/transformers.pdf'))

# %%
len(documents)

# %%
documents[0]

# %%
documents[0].dict().keys()

# %%
documents[0].id_

# %%
documents[0].text

# %%
print(documents[0].text)

# %% [markdown] id="GXQ18XACT334"
# # 1. Vector Store Index

# %% id="91b7xZYA9xjj"
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter

# %% id="f30twJjzoWzy"
vector_index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(
    chunk_size=512,
    chunk_overlap=20,
    separator=" ",
)])

# %%
vector_index.index_id

# %%
vector_index.index_struct.index_id

# %%
vector_index.index_struct.to_dict()

# %%
# Get the nodes_dict from the index struct.
vector_index.index_struct.nodes_dict

# %%
len(vector_index.index_struct.nodes_dict)  # total 28 nodes were created from 15 documents

# %%
# Retrieve a dict mapping of ingested documents and their nodes+metadata.
vector_index.ref_doc_info

# %%
vector_index.vector_store.persist("vector_index.json")

# %%
vector_index.docstore.persist("vector_index_docstore.json")

# %% [markdown]
# ### Vector Store Index as Retriever

# %% id="TqdXql6_UAIU"
vector_retriever = vector_index.as_retriever(similarity_top_k=3)

# %% id="3dA4-LRvUCAj"
nodes = vector_retriever.retrieve("What is the use of positional encodings?")

# %% colab={"base_uri": "https://localhost:8080/"} id="nbcd4s2SUCCz" outputId="d371ff63-f863-4134-dd00-5afe7c64522a"
len(nodes)

# %%
type(nodes)

# %%
nodes[0]

# %%
nodes[0].dict().keys()

# %%
print(nodes[0].get_content())

# %% colab={"base_uri": "https://localhost:8080/"} id="ta2-hh2KUIG8" outputId="e7b35f95-ba85-4455-bdb6-0f6633bd60e4"
for node in nodes:
  print("Node Id:",node.id_)
  print("Metadata:",node.metadata)
  print("Score:",node.get_score())
  print("------------------------------------")

# %% [markdown] id="KUzPVO0L74G6"
# # 2. Summary Index

# %% [markdown]
# - When building the Index, it ingests a collection of documents, splits them into smaller chunks, and then compiles these chunks into a sequential list. 
# - Everything runs locally, without involving an LLM or any embedding model.

# %% [markdown]
# ![summary index.jpg](attachment:50285828-666e-4a47-bec9-7f74060896ac.jpg)

# %% id="Gk13VAty7xUO"
from llama_index.core import SummaryIndex
summary_index = SummaryIndex.from_documents(documents)

# %%
summary_index.index_id

# %%
# Get the index_id from the index struct.
summary_index.index_struct.index_id

# %%
# Get the nodes_dict from the index struct.
summary_index.index_struct.to_dict()

# %%
summary_index.summary

# %%
summary_index.index_struct.nodes

# %%
# print the text present in the Node index 4  >>> 5th Node
node4 = summary_index.index_struct.nodes[4]
print(summary_index.docstore.docs.get(node4).text)

# %%
summary_index.docstore.persist("summary_index_docstore.json")

# %% [markdown] id="1JBk-6Yp8b6t"
# ## Summary Index LLM Retriever

# %% id="sfA-QhGK7xif"
summary_retriever = summary_index.as_retriever(retriever_mode='llm') 

# %% id="mcCAekB777H0"
nodes = summary_retriever.retrieve("What is the use of positional encodings?")

# %%
len(nodes)

# %% colab={"base_uri": "https://localhost:8080/"} id="QHPZ-nnp87z4" outputId="e3eb98d5-7044-434d-8e0c-19778a214b68"
for node in nodes:
  print("Node Id:",node.id_)
  print("Metadata:",node.metadata)
  print("------------------------------------")

# %%
print(nodes[0].get_content())

# %% [markdown] id="PfpT7LTnvbdD"
# ## Summary Index Embedding Retriever

# %% id="91b7xZYA9xjj"
summary_embed_retriever = summary_index.as_retriever(retriever_mode='embedding', similarity_top_k=3)

# %% id="1QDBa6hT-FIk"
nodes = summary_embed_retriever.retrieve("What is the use of positional encodings?")

# %% colab={"base_uri": "https://localhost:8080/"} id="gNScQ9cVR0mG" outputId="524ab0db-df30-402a-d076-6a7a030a491f"
len(nodes)

# %% colab={"base_uri": "https://localhost:8080/"} id="dtZwwmSO-FNc" outputId="abf30719-5a1d-4ab9-9cc7-7d0295e6e188"
for node in nodes:
  print("Node Id:",node.id_)
  print("Metadata:",node.metadata)
  print("Score:",node.get_score())
  print("------------------------------------")

# %% [markdown] id="zzbqXyy0E4e4"
# # 3. Keyword Table Index
#
# ## Keyword Table Simple Retriever

# %% id="FdSwcuvOE5Cv"
from llama_index.core import SimpleKeywordTableIndex

# %% colab={"base_uri": "https://localhost:8080/", "height": 81, "referenced_widgets": ["28c601d47dff40f8b99efb5363b48498", "d16ca29655484909b3b178fc84c112f3", "8ae3bfb0a70d411190bdc37a23ed5ac5", "563ed4ac8e834d9d97d3a0870bf1c14d", "42e7cd4e45ba4fed94960d401ef7ca72", "0b4d88b7c7eb40ad89498904d07d3b2d", "a4d26784a0b544b0b62a4ed3ea466685", "fcc48597122e46d1a7699b56d8a2e070", "63792bbea17d481aa27d6d79faefeb85", "33b55f8997024c1f8f5e71950af18d25", "9312524cad6949c3ae53a6b6098e66fd", "29c62fe2fde543e1bb9fc74b8191d093", "0e15a6e06e154729b5eaabf4d7a1b774", "562215e19296442a9533cb7a94d9c04c", "d9b6488f2a0b45f5bb402194be525d18", "0f19e016f4094c4286bce3f4b3bd9632", "cc8ffcb1e85c490ead6cdf7fb26929fd", "f189d65b733d42f783b8da489023897d", "68caaea8a1d948e4a2e10c0f7c8afd66", "f93ef85f8c7e410da914f9e7ef7e4fb4", "5320257b3bf54e29a1cdf73582d31186", "0e8cd08ce7ff4093851adc9fb8da9299"]} id="HTLMwet5E5Fq" outputId="02d431d8-ad58-4da3-eb73-a5ab52f3c091"
keyword_table_index = SimpleKeywordTableIndex.from_documents(documents, show_progress=True)

# %% id="Mw4GFB60E5Hm"
keyword_table_retriever = keyword_table_index.as_retriever(response_mode='simple')

# %% id="AlK2YUWUGkMw"
nodes = keyword_table_retriever.retrieve("What is the use of positional encodings?")

# %% colab={"base_uri": "https://localhost:8080/"} id="Y4f-_j-gGCWz" outputId="c6a7fa23-d555-434d-f429-c8d90114db9e"
len(nodes)

# %% colab={"base_uri": "https://localhost:8080/"} id="xbI_HLPHZ9xa" outputId="71a1b1bc-813b-4a32-dd8c-b47c92735ccc"
for node in nodes:
  print("Node Id:",node.id_)
  print("Metadata:",node.metadata)
  print("------------------------------------")

# %% [markdown] id="H48QYgqY_EEf"
# # 4. Document Summary Index

# %% [markdown]
# - At its core, the `DocumentSummaryIndex` is designed to optimize information retrieval by summarizing Documents and mapping these summaries to their corresponding Nodes within the Index. 
#
# - This process facilitates efficient data retrieval, using the summaries to quickly identify relevant Documents.

# %% [markdown]
# ![document summary index.jpg](attachment:221fd0fb-feaa-40d4-a69c-31a37b56763b.jpg)

# %% id="irxhzK-t_XKU"
from llama_index.core import DocumentSummaryIndex

# %%
from llama_index.core import get_response_synthesizer

# %% colab={"base_uri": "https://localhost:8080/", "height": 381, "referenced_widgets": ["0791847cc196420ba25c65b7823fe485", "77065e8b8a6c48479491540c574729d5", "fec7029f6d5c48e09ef6e1914bb38373", "0ea368731aa34c579b0df6e04e8d44e9", "d3b616be6d4d45e9b2b31aeab8a89565", "6ee1d71fd13a4206ab1ab566e3e6c0b6", "636698023d8d45b1a8cde4fad9181f82", "378c8b2928ff475dad9baca0c908a271", "bb97d4e0d2c141c48556cfa4149f1739", "97a420d625da46e8adc47ba0ec4b293b", "25f30966ddc14a03ab32cf14673e3887", "ca79fe3f77ea406a8df525cf03ba173f", "5630f88b759449e286a1889a3e0b47f1", "cd5f06076b134908b2cc59168fb1f95d", "d31623dd407e4c2db4262131f3dc82ee", "d0c605e642c54e37893dc9f79c4621a4", "37c3e5cbee1946e6a20962b0bd52b48c", "79ad7071e4de495897b1f33d7d039915", "b1383b78552a46d7981dec6d73be6aea", "13f7fee4abb8443ebccf3114fb13ce59", "023a4e8b8fb64d2ebae3cc00c271e0de", "bbfbcc65912d4f1b91e74ac2519f1ce5", "dddd7eec43834c488d9c6f5565421ddb", "f22508551c0f4606862af1765e1010f6", "a3377d2e45a448dda9f0abdd3caf51ba", "3929d15b0daa47e69c3e3893914d1190", "7edf425c37d34fe0ad42a5e382ce241e", "78c2a7cbf02d40538235539c39f6a105", "a4ed01a78a49456e81749c28988c3b5f", "f6d1f42d73e643799a9a9a5a466f97e4", "23417b1b0c654cf68715d01993f0ab89", "245d1015993e4da98cfd221cb36907fc", "d9c30abb6ebb4c79b30fbb429bdd5782"]} id="4mO5VNnC_XTp" outputId="db919a30-1fcb-4922-c198-e8f72daca4a6"
response_synthesizer = get_response_synthesizer(response_mode="tree_summarize") # "refine", "compact", "accumulate", "compact_accumulate"

doc_summary_index = DocumentSummaryIndex.from_documents(documents, response_synthesizer=response_synthesizer, show_progress=True)

# %%
print(doc_summary_index.get_document_summary(doc_id='20aede31-61a3-4940-b419-acb7f8896895'))

# %%
summary1 = doc_summary_index.get_document_summary(documents[0].doc_id)
summary2 = doc_summary_index.get_document_summary(documents[1].doc_id)
print("\n Summary of the first document: " + summary1)
print("**************************************************")
print("\n Summary of the second document: " + summary2)

# %%
doc_summary_index.index_id

# %%
doc_summary_index.index_struct.to_dict().keys()

# %%
doc_summary_index.index_struct.doc_id_to_summary_id

# %%
doc_summary_index.index_struct.node_id_to_summary_id

# %%
doc_summary_index.index_struct.summary_id_to_node_ids

# %%
doc_summary_index.vector_store.persist("document_summary_vector_store.json")

# %%
doc_summary_index.docstore.persist("document_summary_doc_store.json")

# %% [markdown] id="vL--tmLJAB-n"
# ## Document Summary Index LLM Retriever

# %% id="x1ihRFAH_XWA"
doc_summary_llm_retriever = doc_summary_index.as_retriever(retriever_mode='llm')

# %% id="rhNEROX_-9ba"
nodes = doc_summary_llm_retriever.retrieve("What is the use of positional encodings?")

# %% colab={"base_uri": "https://localhost:8080/"} id="m2VCq92GOdOQ" outputId="fb8102cf-9156-46f8-8612-061bdffcc7a6"
len(nodes)

# %% colab={"base_uri": "https://localhost:8080/"} id="KMb7NQMy-9dw" outputId="8a2685e1-79c7-4ec4-de74-f2cd2a98ab9f"
for node in nodes:
  print("Node Id:",node.id_)
  print("Metadata:",node.metadata)
  print("------------------------------------")

# %%
print(nodes[0].text)

# %% [markdown] id="TYG-R9pRB8OH"
# ## Document Summary Index Embedding Retriever

# %% id="aYGk4WMr-9iL"
doc_summary_embed_retriever = doc_summary_index.as_retriever(retriever_mode='embedding', similarity_top_k=3)

# %% id="9dpWeP6HEtMy"
nodes = doc_summary_embed_retriever.retrieve("What is the use of positional encodings?")

# %%
len(nodes)

# %% colab={"base_uri": "https://localhost:8080/"} id="nastnKLXEtO9" outputId="9646e0f4-4181-43c5-9c65-242ca1c792a3"
for node in nodes:
  print("Node Id:",node.id_)
  print("Metadata:",node.metadata)
  print("------------------------------------")

# %% id="F9Z1FiJdQKJc"
print(nodes[0].text)

# %%
