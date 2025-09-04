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

# %% [markdown] id="5325ac27-38ea-47aa-afef-be4ec4f8f4b9"
# # Auto Merging Retriever
#
# In this notebook, we showcase our `AutoMergingRetriever`, which looks at a set of leaf nodes and recursively "merges" subsets of leaf nodes that reference a parent node beyond a given threshold. This allows us to consolidate potentially disparate, smaller contexts into a larger context that might help synthesis.
#
# You can define this hierarchy yourself over a set of documents, or you can make use of our brand-new text parser: a HierarchicalNodeParser that takes in a candidate set of documents and outputs an entire hierarchy of nodes, from "coarse-to-fine".

# %%
import os 
from dotenv import load_dotenv, find_dotenv

# %%
load_dotenv('/home/santhosh/Projects/courses/Pinnacle/.env')

# %%
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# %% [markdown] id="b9fbdc26"
# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# %% [markdown] id="7e1316ac-84ca-41d0-80f9-d4ef758e653c"
# ## Load Data
#
# Let's first load the Llama 2 paper: https://arxiv.org/pdf/2307.09288.pdf. This will be our test data.

# %% id="80372299-ab32-4ddd-9b88-05c877120c17"
# !mkdir -p 'data/'
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"

# %% id="5f9c5d99-bd0e-4b26-b816-9f5ad29df3c8"
from pathlib import Path
from llama_index.readers.file import PyMuPDFReader

# %%
from llama_index.core import SimpleDirectoryReader

# %%
docs0 = SimpleDirectoryReader(input_files=["./data/llama2.pdf"]).load_data()

# %% id="723f1f02-2157-4166-b013-90e627c76530"
loader = PyMuPDFReader()
# docs0 = loader.load_data(file=Path("./data/llama2.pdf"))
docs0 = loader.load(file_path=Path("./data/llama2.pdf"))

# %% [markdown] id="ff7a8552-f347-45b0-b4a0-4f9b32be57ac"
# By default, the PDF reader creates a separate doc for each page.
# For the sake of this notebook, we stitch docs together into one doc.
# This will help us better highlight auto-merging capabilities that "stitch" chunks together later on.

# %% id="a75c4217-ab50-417f-a8ed-3b746a9956c8"
from llama_index.core import Document

doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]

# %% [markdown] id="724fe6f1-80e1-4ac5-bd99-8b9b8d15bddd"
# ## Parse Chunk Hierarchy from Text, Load into Storage
#
# In this section we make use of the `HierarchicalNodeParser`. This will output a hierarchy of nodes, from top-level nodes with bigger chunk sizes to child nodes with smaller chunk sizes, where each child node has a parent node with a bigger chunk size.
#
# By default, the hierarchy is:
# - 1st level: chunk size 2048
# - 2nd level: chunk size 512
# - 3rd level: chunk size 128
#
#
# We then load these nodes into storage. The leaf nodes are indexed and retrieved via a vector store - these are the nodes that will first be directly retrieved via similarity search. The other nodes will be retrieved from a docstore.

# %% id="45e783f5-a323-4f51-ae9a-4b71b00e5e11"
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter

# %% id="2c3947df-25c2-4254-a3d4-381d136f3f77"
node_parser = HierarchicalNodeParser.from_defaults()

# %% id="2162b309-dfc5-484b-a31c-24f705316f10"
nodes = node_parser.get_nodes_from_documents(docs)

# %% id="a9b5bc9b-389d-47db-a41c-3eb5b3d38ac5" outputId="74627310-8220-4ec4-e4f2-bd6c21746bab"
len(nodes)

# %% [markdown] id="a7456b70-1803-4786-86d5-26e202e0f318"
# Here we import a simple helper function for fetching "leaf" nodes within a node list.
# These are nodes that don't have children of their own.

# %% id="7299ca7e-09b6-432f-a277-aae9eca0522a"
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes

# %% id="faeb37a8-aea9-4ee8-b6c0-3b2f188d244e"
leaf_nodes = get_leaf_nodes(nodes)

# %% id="7c33b5a8-4d9f-481e-8616-fc8717900159" outputId="e98e34b1-d89e-406a-d224-b06fe6cbf16a"
len(leaf_nodes)

# %% id="b28b0aff-db6e-495e-8c58-36db29edb45b"
root_nodes = get_root_nodes(nodes)

# %% [markdown] id="c36ec940-8af7-45f5-9994-919d57583c24"
# ### Load into Storage
#
# We define a docstore, which we load all nodes into.
#
# We then define a `VectorStoreIndex` containing just the leaf-level nodes.

# %% id="27c8f2cd-3e04-4feb-937b-b9ee33e1c2fd"
# define storage context
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

docstore = SimpleDocumentStore()

# insert nodes into docstore
docstore.add_documents(nodes)

# define storage context (will include vector store by default too)
storage_context = StorageContext.from_defaults(docstore=docstore)

# %% id="827ece8e-7a4b-4ee1-8ee2-3433d7f2072a"
## Load index into vector index
from llama_index.core import VectorStoreIndex

base_index = VectorStoreIndex(
    leaf_nodes,
    embed_model = OpenAIEmbedding(model='text-embedding-3-small'),
    storage_context=storage_context,
)

# %% [markdown] id="05d84c19-c9ac-4294-a000-264c3c02427b"
# ## Define Retriever

# %% id="e61682a0-dd3c-400b-8734-35d5d0a98252"
from llama_index.core.retrievers.auto_merging_retriever import AutoMergingRetriever

# %% id="f96fd0bc-c6c0-4073-a692-d1803cf4289f"
base_retriever = base_index.as_retriever(similarity_top_k=6)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)

# %% id="62f655cd-4195-4398-80e5-5aa561982d25" outputId="099c7535-159e-4b42-981d-2fab54c4120b"
# query_str = "What were some lessons learned from red-teaming?"
# query_str = "Can you tell me about the key concepts for safety finetuning"
query_str = (
    "What could be the potential outcomes of adjusting the amount of safety data used in the RLHF stage?"
)

base_nodes = base_retriever.retrieve(query_str)

nodes = retriever.retrieve(query_str)

# %% id="62f655cd-4195-4398-80e5-5aa561982d25" outputId="099c7535-159e-4b42-981d-2fab54c4120b"
# query_str = "What were some lessons learned from red-teaming?"
# query_str = "Can you tell me about the key concepts for safety finetuning"
query_str = (
    "What could be the potential outcomes of adjusting the amount of safety data used in the RLHF stage?"
)

base_nodes = base_retriever.retrieve(query_str)

nodes = retriever.retrieve(query_str)

# %% id="62f655cd-4195-4398-80e5-5aa561982d25" outputId="099c7535-159e-4b42-981d-2fab54c4120b"
# query_str = "What were some lessons learned from red-teaming?"
# query_str = "Can you tell me about the key concepts for safety finetuning"
query_str = (
    "What could be the potential outcomes of adjusting the amount of safety data used in the RLHF stage?"
)

base_nodes = base_retriever.retrieve(query_str)

nodes = retriever.retrieve(query_str)

# %% id="77eabc56-2009-4504-8832-b6d857bd43a4" outputId="4729aed3-28ad-4cdf-d0e2-c1fb9b88507a"
len(nodes)

# %% id="3a82690a-263a-4e48-ab27-b161e72cb983" outputId="72f1ac53-6024-49a9-9eee-fbd1f36dea7e"
len(base_nodes)

# %% id="0d482b22-fd38-476b-821f-0c77564815c3" outputId="6bf61dad-bcb8-4d8c-8e85-9f5a26a31e29"
from llama_index.core.response.notebook_utils import display_source_node

for node in nodes:
    display_source_node(node, source_length=10000)

# %% id="e4dd58db-4b12-49dc-b42f-8a0ee746f5c9" outputId="082057d1-6f7a-4166-9407-eee463d45ac0"
for node in base_nodes:
    display_source_node(node, source_length=10000)

# %% [markdown] id="08f62e2c-4def-402e-8904-47f34d12c2fb"
# ## Plug it into Query Engine

# %% id="5d3ce9ec-f6cd-475b-94fa-3e8df81ab824"
from llama_index.core.query_engine import RetrieverQueryEngine

# %% id="f106e1bb-58bc-48bf-a46b-e527339f83c5"
query_engine = RetrieverQueryEngine.from_args(retriever)
base_query_engine = RetrieverQueryEngine.from_args(base_retriever)

# %% id="94a85854-ca04-41ed-9f44-b6dce1e513e1" outputId="9df10b31-5035-4ada-9a16-71b6e73063e4"
response = query_engine.query(query_str)

# %% id="8b334b7b-fcb8-4057-a418-b8d8c425ad14" outputId="da7e090a-baad-4398-b3dd-9ce92dd4cca1"
print(str(response))

# %% id="1c38a124-5279-4a43-a2fe-ed2cbce9bd66"
base_response = base_query_engine.query(query_str)

# %% id="5c2910e5-1a45-4de5-8035-5b5a47125d81" outputId="c4936ebf-068e-4bbf-97e0-06701ade526c"
print(str(base_response))

# %% [markdown] id="f84be450-c036-4cca-bf94-ccfc18e5d52a"
# ## Evaluation
#
# We evaluate how well the hierarchical retriever works compared to the baseline retriever in a more quantitative manner.
#
# **WARNING**: This can be *expensive*, especially with GPT-4. Use caution and tune the sample size to fit your budget.

# %% id="cb5a6511-5756-4cdb-933e-f530f0c40bc3"
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.llms.openai import OpenAI
import nest_asyncio

nest_asyncio.apply()

# %%
gpt4 = OpenAI(model='gpt-4o')

# %% id="914e3056-d9e3-42a0-9600-a66ae6a9f075"
# NOTE: run this if the dataset isn't already saved
dataset_generator = RagDatasetGenerator(
    root_nodes[:2],
    llm=gpt4,
    show_progress=True,
    num_questions_per_chunk=3,
)

# %% id="d9569b69-d9bf-4b85-a1b3-3ff4ef6619b8"
eval_dataset = await dataset_generator.agenerate_dataset_from_nodes()

# %% id="e5b3ba74-0092-4906-88cc-638fa304c97b"
eval_dataset.save_json("data/llama2_eval_qr_dataset.json")

# %% id="87f30894-ba65-4af7-9380-210f6a5b2de4"
# optional
eval_dataset = LabelledRagDataset.from_json(
    "data/llama2_eval_qr_dataset.json"
)

# %% [markdown] id="2d793ae5-80be-41ff-8b1f-e92ea27b2a8b"
# ### Compare Results
#
# We run evaluations on each of the retrievers: correctness, semantic similarity, relevance, and faithfulness.

# %% id="90ca35cf-e659-4a1e-8561-d07d50972b3a"
import asyncio
import nest_asyncio

nest_asyncio.apply()

# %% id="f6814643-bdb2-47cd-a8a5-69a1bdfdda30"
from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    RelevancyEvaluator,
    FaithfulnessEvaluator,
    PairwiseComparisonEvaluator,
)


from collections import defaultdict
import pandas as pd

# %%
gpt4 = OpenAI(temperature=0, model="gpt-4o")

# %%
# NOTE: can uncomment other evaluators
evaluator_c = CorrectnessEvaluator(llm=gpt4)
evaluator_s = SemanticSimilarityEvaluator(embed_model=OpenAIEmbedding(model='text-embedding-3-small'))
evaluator_r = RelevancyEvaluator(llm=gpt4)
evaluator_f = FaithfulnessEvaluator(llm=gpt4)
pairwise_evaluator = PairwiseComparisonEvaluator(llm=gpt4)

# %% id="ae472816-5927-4f67-9105-fd0ba0c60f49"
from llama_index.core.evaluation.eval_utils import get_responses, get_results_df
from llama_index.core.evaluation import BatchEvalRunner

# %%
eval_qs = [example.query for example in eval_dataset.examples]
ref_response_strs = [example.reference_answer for example in eval_dataset.examples]

# %% id="a7302e7b-6b3e-4d25-874b-94ffe944b527"
pred_responses = get_responses(eval_qs, query_engine, show_progress=True)

# %% id="0bce562f-6c42-446a-b991-f208ca9f55cb" outputId="02cc7dce-adc0-4229-edce-ffb7139f9886"
base_pred_responses = get_responses(
    eval_qs, base_query_engine, show_progress=True
)

# %% id="3428afe0-b5e4-4604-b7ce-270f449766cb"
import numpy as np

pred_response_strs = [str(p) for p in pred_responses]
base_pred_response_strs = [str(p) for p in base_pred_responses]

# %% id="b0a86029-bf5e-4f26-afdf-a9406bada315"
evaluator_dict = {
    "correctness": evaluator_c,
    "faithfulness": evaluator_f,
    "relevancy": evaluator_r,
    "semantic_similarity": evaluator_s,
}
batch_runner = BatchEvalRunner(evaluator_dict, workers=2, show_progress=True)

# %% id="9abccb17-cf7d-4f59-b849-a4d5581df7a9"
eval_results = await batch_runner.aevaluate_responses(
    eval_qs, responses=pred_responses, reference=ref_response_strs
)

# %% id="7c5df1ec-408a-4012-93c7-a0151fa92b9e"
base_eval_results = await batch_runner.aevaluate_responses(
    eval_qs, responses=base_pred_responses, reference=ref_response_strs
)

# %% id="ea90f363-71d7-404c-9d4d-f6eea386e59f" outputId="f105bf85-7fa9-4402-d6e5-327790919652"
results_df = get_results_df(
    [eval_results, base_eval_results],
    ["Auto Merging Retriever", "Base Retriever"],
    ["correctness", "relevancy", "faithfulness", "semantic_similarity"],
)
display(results_df)

# %% [markdown] id="be393093-9a6d-46d1-9d18-d17e92523200"
# **Analysis**: The results are roughly the same.
#
# Let's also try to see which answer GPT-4 prefers with our pairwise evals.
