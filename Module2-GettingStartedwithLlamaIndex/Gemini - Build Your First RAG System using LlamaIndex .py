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

# %% [markdown] id="ACfLrpwlo8Nl"
# # Build Your First RAG System
#
# 1. Data Ingestion.
# 2. Indexing.
# 3. Retriever.
# 4. Response Synthesizer.
# 5. Querying.

# %% [markdown]
# ## Install Required packages

# %% [markdown]
# Download the required packages by executing the below commands in either Anaconda Prompt (in Windows) or Terminal (in Linux or Mac OS)

# %% [markdown] colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9291, "status": "ok", "timestamp": 1703361268107, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="Elvu26cIedWC" outputId="529ae34f-17a9-4724-b391-f1803279dfed"
# pip install llama-index

# %% [markdown]
# ## Environment Variables

# %% [markdown]
# It is recommonded to store the API keys in a '.env' file, separate from the code.
# Plesae follow the below steps.
# 1. Create a text file with the name '.env'
# 2. Enter your api key in this format OPENAI_API_KEY='sk-e8943u9ru4982............'
# 3. Save and close the file

# %% [markdown]
# Then, as shown below you can provide the path of the '.env' file to 'load_dotenv' method.
# This will load any API keys stored in the '.env' file.

# %% [markdown]
# ## Start

# %% id="cKlax-updNW-"
import os

# %%
from dotenv import load_dotenv, find_dotenv

# %%
load_dotenv('/home/santhosh/Projects/courses/Pinnacle/.env')

# %% [markdown]
# This setup ensures that our API key remains secure and easily configurable. Always remember to keep your `.env` file secure and avoid including it in version control."
#

# %% [markdown] id="yLtBXZ0xDtmQ"
# # Stage 1: Data Ingestion
#
# ## Data Loaders
#

# %% [markdown]
# We start by loading the data from a PDF file. For this, we will use the SimpleDirectoryReader class from LlamaIndex.

# %% id="gGfPPk4gBAkQ"
from llama_index.core import SimpleDirectoryReader

# %%
documents = SimpleDirectoryReader(input_files=['data/transformers.pdf']).load_data()

# %% [markdown]
# We can then check the type of the `documents` variable and the total number of pages read from the PDF:

# %%
# Check the datatype and length of the loaded documents
type(documents)

# %%
# total number of pages read from the PDF
len(documents)

# %%
documents[0]

# %% [markdown]
# **To understand the structure of the loaded documents, let's retrieve the first document, which corresponds to the first page of the PDF:**
#

# %%
# Retrieve the first document (essentially the first page in the PDF)
documents[0]

# %% [markdown]
# We can also access specific attributes of the document, such as its ID and metadata:

# %%
# Get the ID of the first document
documents[0].id_

# %%
documents[0].doc_id

# %%
# Get the metadata of the first document
documents[0].metadata

# %%
# Get the text content of the first document
print(documents[0].text)

# %% [markdown] id="zv9VQB-EdsEd"
# ## Embedding Model

# %% [markdown]
# Next, we need to prepare our document for embedding and interaction with a large language model. We will use the OpenAI API for this purpose.

# %% id="RTOBfe1hc2zu"
# Embedding Model
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding

# %%
# Initialize the embedding model
embed_model = GeminiEmbedding(model="models/embedding-004")

# %% [markdown] id="5UD_RkiXf7Cm"
# ## LLM

# %% [markdown]
# Similarly, let's set up our large language model (LLM):

# %% id="6q6O3wusigcW"
# LLM
from llama_index.llms.gemini import Gemini

# %%
# Initialize the large language model
llm = Gemini(model= "models/gemini-1.5-pro")

# %% [markdown] id="vseCdqiFj7W0"
# # Stage 2: Indexing

# %% id="T9NxcrBpeprP"
# Indexing
from llama_index.core import VectorStoreIndex

# %% [markdown]
# Here, we use the `VectorStoreIndex` class to create an index from the loaded documents. We pass the document chunks, embedding model, and LLM to the `from_documents` method.

# %%
# Create an index from the documents using the embedding model and LLM
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# %% [markdown] id="oA9czIv0sqe_"
# # Stage 3: Retrieval

# %% [markdown]
# Finally, we set up a retriever to query our indexed documents. This allows us to retrieve relevant information based on our queries.

# %% id="8-E66LtRjgT4"
# Setting up the Index as Retriever
retriever = index.as_retriever()

# %% [markdown]
# The `as_retriever` method converts our index into a retriever, and the `retrieve` method allows us to query the index.

# %% id="foOrz7q-oAJl"
# Retrieve information based on the query "What are Transformers?"
retrieved_nodes = retriever.retrieve("What is self attention?")

# %% [markdown]
# We can check the metadata of the retrieved nodes to understand the source of the information:

# %% [markdown]
# The metadata provides details such as the page label, file name, file path, file type, and other relevant information.

# %%
# Get the metadata of the first retrieved node
retrieved_nodes[0].metadata

# %% [markdown]
# let's access the ID of the first retrieved node, which is a unique identifier for the first node:

# %%
# Access the ID of the first retrieved node
retrieved_nodes[0].id_

# %% [markdown]
# Similarly, we can access the node_id attribute, which typically holds the same value:

# %%
# Access the node_id of the first retrieved node
retrieved_nodes[0].node_id

# %% [markdown]
# Next, let's explore the `node` attribute of the retrieved node. This attribute contains a `TextNode` object, which holds all the relevant information extracted during the retrieval process: The `TextNode` object includes various details such as metadata and text content.

# %% colab={"base_uri": "https://localhost:8080/", "height": 157} executionInfo={"elapsed": 401, "status": "ok", "timestamp": 1703361342847, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="EXpkVs2RoHsA" outputId="d3f9b5b0-d90c-43b6-b194-8d05c249aa97"
# Access the full node object of the first retrieved node
retrieved_nodes[0].node

# %% [markdown]
# We can also extract and inspect the text content of this node to understand the retrieved information better:

# %% colab={"base_uri": "https://localhost:8080/", "height": 157} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1703361344340, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="c5KctGWPLi7u" outputId="e463cbb7-d169-456c-bccb-ab15004f711a"
# Access the text content of the first retrieved node
print(retrieved_nodes[0].text)

# %%
retrieved_nodes[1].metadata

# %%
print(retrieved_nodes[1].text)

# %% [markdown] id="Ty19sHbWxoEu"
# # Stage 4: Response Synthesis
#

# %% [markdown]
# We need to synthesize responses from our large language model (LLM). For this, we use the `get_response_synthesizer` function:

# %% id="TnLdxijaxw80"
from llama_index.core import get_response_synthesizer

# %% [markdown]
# Here, the `get_response_synthesizer` function takes our LLM as an argument and returns a synthesizer object that will help generate coherent responses to our queries.

# %%
# Initialize the response synthesizer with the LLM
response_synthesizer = get_response_synthesizer(llm=llm)

# %% [markdown] id="orz-nHJYyz0u"
# ## Stage 5: Query Engine

# %% [markdown]
# Next, we set up a query engine. This engine will allow us to query our indexed documents and receive synthesized responses from the LLM:

# %% id="EiHo7R3K0OH3"
# Create a query engine using the index, LLM, and response synthesizer
query_engine = index.as_query_engine(llm=llm, response_synthesizer=response_synthesizer)

# %% [markdown]
# We use the `as_query_engine` method from our index object to create a query engine, passing the LLM and response synthesizer as arguments.
#
# With our query engine ready, we can now query the LLM using natural language:
#

# %% id="dTCGOKvI1Zj_"
# Query the LLM using the query engine
response = query_engine.query("What is self attention?")  

# %% [markdown]
# In this command, we query the LLM with the question "What are Transformers?" and store the response in the `response` variable.
#
# To view the response generated by the LLM, we can access the `response` attribute:
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1703361376718, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="mTgwMcaJ1nhT" outputId="e571e28d-711d-4c45-e7f7-0fd882677ee6"
# View the response from the LLM
response.response 

# %% [markdown]
# This returns the synthesized answer to our query.
#
# We can further analyze the response by checking its length and inspecting the source nodes used to generate it:
#

# %% [markdown]
# These commands provide the length of the response and the number of source nodes, respectively.

# %%
# Check the length of the response
len(response.response) # number of characters in the response

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1703361403705, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="aRMcVB1nQBbp" outputId="f84a6390-2f65-4ada-d322-5a63097a5187"
# Check the number of source nodes
len(response.source_nodes)  # list of 2 nodes

# %%
# Access the ID and metadata of the first source node
response.source_nodes[0].id_

# %%
# Access the ID and metadata of the second source node
response.source_nodes[0].metadata

# %%
response.source_nodes[1].id_

# %%
response.source_nodes[1].metadata

# %% [markdown] id="VRUMuoRK7qvt"
# # End to End RAG Pipeline

# %% [markdown]
# In this final section, we will integrate everything we have learned to create a complete end-to-end Retrieval-Augmented Generation (RAG) pipeline. This pipeline will read documents, index them, and allow us to query the indexed data using a large language model (LLM).
#
# Let's walk through the entire process step by step:

# %% [markdown]
# - First, we import the necessary libraries and load our documents from a specified directory. We use the `SimpleDirectoryReader` class from LlamaIndex to read all documents in the 'data' directory:
#
#
# - The `SimpleDirectoryReader` reads the documents in the 'data' directory and stores them in the `documents` variable.
#
# - Next, we initialize our large language model (LLM) and embedding model. For this demonstration, we assume that these models have already been initialized and are available as `llm` and `embed_model`:
#
# - With our documents and models ready, we proceed to create an index. This index will facilitate efficient retrieval of information from our documents. Here, we use the `VectorStoreIndex` class to create an index from the loaded documents, embedding model, and LLM.
#
# - We then set up a query engine that will allow us to query the indexed documents using natural language. The query engine is created from our index and LLM:
#
# - Finally, we use the query engine to ask a question and receive a response from the LLM. In this example, we query the different types of Transformer models:
#
# - The `query` method sends the question to the LLM, which retrieves relevant information from the indexed documents and synthesizes a response. The response is then printed to the console.
#
#
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1806, "status": "ok", "timestamp": 1703361456294, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="HhAb3o0l7wwD" outputId="e683a5ab-23ac-486f-ba11-0d4b51bd8499"
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Load data from the specified directory
documents = SimpleDirectoryReader("data").load_data()

# Initialize LLM and embedding model (assumed to be pre-initialized)
llm = llm
embed_model = embed_model

# Create an index from the documents using the embedding model and LLM
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, llm=llm)

# Create a query engine from the index and LLM
query_engine = index.as_query_engine(llm=llm)

# Query the LLM and print the response
print(query_engine.query("What are the different types of Transformer Models?").response)

# %%
print(query_engine.query("Why do we need positional encodings in transformer?").response)

# %%
print(query_engine.query("What are Encoder and Decoder blocks in transformer?").response)

# %%
query = "If I want to generate document embeddings, then which type of Transformer Architecture I must choose?"
print(query_engine.query(query).response)

# %%
query = """If I want to generate document embeddings, 
then which type of Transformer Architecture I must choose among Encoders, Decoders or Encoder-Decorder?"""

print(query_engine.query(query).response)

# %%

# %% [markdown]
# By following these steps, we have created a fully functional end-to-end RAG pipeline. This pipeline can ingest documents, index them, and answer natural language queries using a powerful combination of LlamaIndex and OpenAI's models. This demonstrates the practical application of RAG systems in extracting and synthesizing information from large datasets.
#

# %%
