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

# %% [markdown] id="dLXyxks9D-EA"
# In this project we will look into Evaluating different indices on the `BlockchainSolanaDataset` dataset.

# %% [markdown]
# <b> Evaluation of RAG can be costly GPT-4 is being used. Please keep track of the cost. You can try to run on lesser data to reduce cost.

# %% executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1703447936066, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="h_mRdWvUeOD7"
import nest_asyncio
nest_asyncio.apply()

# %% id="cKlax-updNW-"
import os

# %%
from dotenv import load_dotenv, find_dotenv

# %%
load_dotenv('/home/santhosh/Projects/courses/Pinnacle/.env')

# %% [markdown] id="CChQ98mgWGcC"
# #### Download Evaluation Dataset of `BlockchainSolanaDataset` from LlamaDatasets which is based on [Blockchain to Solana Paper](https://arxiv.org/pdf/2207.05240.pdf)

# %%
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.packs.rag_evaluator import RagEvaluatorPack
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# %% [markdown]
# Download the required files from the below link if needed and move them to the folders mentioned below in the code

# %% [markdown]
# https://github.com/run-llama/llama-datasets/tree/main/llama_datasets/blockchain_solana

# %%
rag_dataset = LabelledRagDataset.from_json("./data/rag_dataset.json")
documents = SimpleDirectoryReader(input_dir="./data/source_files").load_data()

# %% executionInfo={"elapsed": 11683, "status": "ok", "timestamp": 1703447955117, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="7smekBCXWS3X"
eval_queries = [example.query for example in rag_dataset.examples]
eval_answers = [example.reference_answer for example in rag_dataset.examples]

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 40, "status": "ok", "timestamp": 1703447955125, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="O1_ooUWgEhd9" outputId="34dc0f8f-e7c0-4192-b514-9cbb7a210904"
documents[0]

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 17, "status": "ok", "timestamp": 1703447955125, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="xvlAnBSJEdxv" outputId="fc64bb21-b657-4ec3-d89d-c927a5c0ca3e"
len(eval_queries)

# %% colab={"base_uri": "https://localhost:8080/", "height": 52} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1703447955125, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="TqZDxIC_Efgp" outputId="c3bdd673-c97f-465e-cadc-7aa763bc1a45"
eval_queries[0]

# %% colab={"base_uri": "https://localhost:8080/", "height": 104} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1703447955126, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="YeDrbzwUEg6N" outputId="e782313f-d3b8-453a-af04-e88a4cafe42f"
eval_answers[0]

# %% [markdown] id="tqxH7IhQEsIA"
# # LLM

# %% executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1703447972838, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="6gtoOK84oSEe"
from llama_index.llms.openai import OpenAI
gpt4o_mini = OpenAI(model="gpt-4o-mini", temperature=0.1)

# %% [markdown] id="bUU4g-ZsEth7"
# # Embedding Model

# %% executionInfo={"elapsed": 6429, "status": "ok", "timestamp": 1703447984402, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="ZrMFq7aaocKj"
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# %% [markdown] id="hPy2Sc9oFqtM"
# # Build Vector Store Index

# %% colab={"base_uri": "https://localhost:8080/", "height": 81, "referenced_widgets": ["2d11b1ce22a9422eb82cf7fbdb152e6b", "56b324438d404b40a9352823b80158dc", "1d3386753fab488ca311d36e47d9341e", "00dcc17ce3314bca8e00ca72b8a134d0", "fe44f06874c04f01a2859d22c8e54b94", "8aff5757a6104052b75d4f52377775cc", "fb88b5f1cd6e449aab8fa6c322a9ac10", "e7ccaeb1d43b4794974d4fdb824cbe08", "717b8dec8572452b8a35b252bdcdf428", "adeeeab069334a02b70b2a0f1e704588", "6c3fd6c7ec3b438ebbbdfc8e797d4583", "dacc78548f6e4bc9ae37cada2cd342b7", "477f102ec1044dbba889c6a44d0b8197", "7d2b55ba2b3e43578e41820be758e610", "9759ca860955460fb9cbf540c1479d02", "386889e72e31483c96a775750312233c", "5aae452ea6714bbbbab440e6299afaa1", "c65aa77dd8644b6b8ac5b7123f59b360", "fbaf749231e44f9ea77b3096b9d2ae48", "8d4cf1669fac4169ab942904568c204c", "75a8315a96f24e4ba7b0b037418015a6", "25de09e26c8a476e9763bc325e0346d6"]} executionInfo={"elapsed": 31994, "status": "ok", "timestamp": 1703448029709, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="Q9DloDHrj5Bh" outputId="0c7ae2fc-2e47-43e3-dff3-0ff47144c8e8"
from llama_index.core import VectorStoreIndex
vector_store_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, show_progress=False)

# %% executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1703448029709, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="0wNL3I2pF9Wm"
vector_store_query_engine = vector_store_index.as_query_engine()

# %% [markdown] id="hbi7xqlkU1I1"
# # Build Keyword Table Index

# %% colab={"base_uri": "https://localhost:8080/", "height": 81, "referenced_widgets": ["0f67ffc4f98141f2b93f63458b3610b6", "1670a60c41b34e9c8bc16dd0b047e798", "67c3853572ce4f7d86f10ead96541002", "56101942bac94f0e869e64633db6c13a", "832be85518bd48c088581a81c59b4aee", "cecad5376cdc4657bdb23cb51969700c", "3cbb486c439a42c68de0ee022532a09e", "9481e3eb07aa405cbe6128a4cb5eae81", "59ebe474fd4b46bc8aa1786dc89b0a47", "b5be027fc3cd4952b6844ffbd4217b30", "ed61278f394f430badd102e4c9c746fa", "a625c10b47034295b212f049017f5e0d", "70c7aa4e3c194e94bfd4a616a5d59670", "87308688616c4a22801e12bdb21033b3", "3dfdf1b86bfb487d9bfe6ac554aba89b", "cd48eaa105734d0bbd9a5aef580886ea", "6571791b0d2849f3b28362d678397a6c", "d7f6f87e96014572a45b7b43abb1d2ad", "5e9b559fbfc146b8863f0e06bda8ea65", "413bd713b51b46e29f7ef8ac2744c1b7", "e5d731a3f99d4137bfd22d20962ad36d", "618cc01c1ccb4808be9c0e3dc0e552b2"]} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1703448029709, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="sX8uasn1U8sK" outputId="af8a41ce-b5cb-4463-a632-ec77dd13bf19"
from llama_index.core.indices import SimpleKeywordTableIndex
keyword_table_index = SimpleKeywordTableIndex.from_documents(
    documents,
    embed_model=embed_model, llm=gpt4o_mini,
    show_progress=False
)

# %% executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1703448029709, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="LG0uZHUgU8xt"
keyword_table_query_engine = keyword_table_index.as_query_engine()

# %% [markdown] id="FzNyTN9RZUUC"
# # Download `RagEvaluatorPack`

# %%
from llama_index.packs.rag_evaluator import RagEvaluatorPack

# %%
rag_evaluator_pack = RagEvaluatorPack(
    rag_dataset=rag_dataset,
    query_engine=vector_store_query_engine
)

# %% [markdown] id="Xy7iIpCUd5MC"
# #Evaluating Vector Store Index

# %% [markdown] id="vYIKG_qBZ89S"
# Compute the metrics for the responses generated by vector store retriever

# %%
gpt4 = OpenAI(model='gpt-4o')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 222676, "status": "ok", "timestamp": 1703448272602, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="3kf7YPgq5iXp" outputId="c6ba0fa2-40a7-4b68-adb3-f317e5c0a421"
rag_evaluator_pack = RagEvaluatorPack(
    rag_dataset=rag_dataset,
    query_engine=vector_store_query_engine,
    judge_llm=gpt4
)

vector_benchmark_df = await rag_evaluator_pack.arun(
    batch_size=10,  # batches the number of openai api calls to make
    sleep_time_in_seconds=1,  # seconds to sleep before making an api call
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 31, "status": "ok", "timestamp": 1703448272602, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="DmzJvf9WC440" outputId="4056d128-2dce-4064-9be6-4a55f17b082c"
vector_benchmark_df.columns = ['VectorStore Index']

vector_benchmark_df

# %% [markdown] id="qOANLD31eb6R"
# #Evaluating Keyword Table Index

# %% [markdown] id="KxE4qiCjccCF"
# Compute the metrics with Keyword Table Index

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 259171, "status": "ok", "timestamp": 1703448531761, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="YqYGdgML6QTd" outputId="5645c05f-f36a-437a-b13d-57d7e4d8bfd3"
rag_evaluator_pack = RagEvaluatorPack(
    rag_dataset=rag_dataset,
    query_engine=keyword_table_query_engine,
    judge_llm=gpt4
)

keyword_table_benchmark_df = await rag_evaluator_pack.arun(
    batch_size=10,  # batches the number of openai api calls to make
    sleep_time_in_seconds=1,  # seconds to sleep before making an api call
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 32, "status": "ok", "timestamp": 1703448531761, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="qmANsLnyCyl3" outputId="5f84feef-b5b3-407f-82a1-d94a6ed084ea"
keyword_table_benchmark_df.columns = ['Keyword Table Index']

keyword_table_benchmark_df

# %% [markdown] id="-AMkexjnfrXp"
# # Display results

# %% executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1703448531761, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="6FIv4ZD4fSkd"
import pandas as pd

results_df = pd.concat([vector_benchmark_df.T, keyword_table_benchmark_df.T])


# %% colab={"base_uri": "https://localhost:8080/", "height": 125} executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1703448531761, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="c8b8OOuPjvKk" outputId="25962983-6b74-4961-89a2-5d6d73cfb6e0"
results_df
