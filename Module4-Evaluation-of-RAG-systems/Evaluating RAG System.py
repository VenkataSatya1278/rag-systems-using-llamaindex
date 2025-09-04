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

# %% [markdown] id="QYkWSVQDPckV"
# # Evaluation
#
# Evaluation and benchmarking play a pivotal role in the development of LLM Applications. For optimizing the performance of applications such as RAG (Retrieval Augmented Generation), a robust measurement mechanism is indispensable.
#
# LlamaIndex offers vital modules tailored to assess the quality of generated outputs. Additionally, it incorporates specialized modules designed specifically to evaluate content retrieval quality. LlamaIndex categorizes its evaluation into two primary types:
#
# *   **Response Evaluation**
# *   **Retrieval Evaluation**
#
# [Documentation
# ](https://gpt-index.readthedocs.io/en/latest/core_modules/supporting_modules/evaluation/root.html)

# %% [markdown] id="oTMyT_qQSH0L"
# # Response Evaluation
#
# Evaluating results from LLMs is distinct from traditional machine learning's straightforward outcomes. LlamaIndex employs evaluation modules, using a benchmark LLM like GPT-4, to gauge answer accuracy. Notably, these modules often blend query, context, and response, minimizing the need for ground-truth labels.
#
# The evaluation modules manifest in the following categories:
#
# *   **Faithfulness:** Assesses whether the response remains true to the retrieved contexts, ensuring there's no distortion or "hallucination."
# *   **Context Relevancy:** Evaluates the relevance of both the retrieved context and the generated answer to the initial query.
# *   **Correctness:** Determines if the generated answer aligns with the reference answer based on the query (this does require labels).
#
# Furthermore, LlamaIndex has the capability to autonomously generate questions from your data, paving the way for an evaluation pipeline to assess the RAG application.

# %% [markdown]
# <b> Evaluation of RAG can be costly GPT-4 is being used. Please keep track of the cost. You can try to run on lesser data to reduce cost.

# %%
import warnings
warnings.filterwarnings("ignore")

# %% executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1703533305432, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="4fTQJZDiZtIR"
# attach to the same event-loop
import nest_asyncio

nest_asyncio.apply()

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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3898, "status": "ok", "timestamp": 1703533309324, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="O384ocD_OjDG" outputId="edf1b8d5-00c3-42bc-dbf2-b3bc0bb6da88"
import logging
import sys
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# %%
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Response,
)

# %%
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    RetrieverEvaluator,
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset
)

# %%
from llama_index.llms.openai import OpenAI

# %%
from llama_index.core.llama_dataset.generator import RagDatasetGenerator

# %% id="cKlax-updNW-"
import os

# %%
from dotenv import load_dotenv, find_dotenv
# load_dotenv('D:/.env')
# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# %%
load_dotenv('D:/Training/FAA-Training/Beyond-the-Prompt-Practical-RAG-for-Real-World-AI/RAG-systems-using-LlamaIndex/RAG-System-Using-LamaIndex/.env')

# %% [markdown] id="CChQ98mgWGcC"
# ### Download Data

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 775, "status": "ok", "timestamp": 1703533310092, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="7smekBCXWS3X" outputId="8b04510c-e58b-481f-8f64-8076330628d0"
# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/9607a05a923ddf07deee86a56d386b42943ce381/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# %% [markdown] id="uNfuJB0xXKw8"
# ### Load Data

# %% executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1703533310092, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="hIz7x-91VyuY"
reader = SimpleDirectoryReader("./data/paul_graham/")
documents = reader.load_data()

# %% [markdown] id="mVy40TPDXQLN"
# ### Generate Questions

# %% id="4iCdPoP8XMY6"
data_generator = RagDatasetGenerator.from_documents(documents, llm=OpenAI(temperature=0, model="gpt-4o-mini"),
                                                   num_questions_per_chunk=2)

# %%
eval_dataset = data_generator.generate_dataset_from_nodes()

# %%
eval_dataset.examples[0].query

# %%
eval_dataset.examples[0].reference_answer

# %%
eval_questions = [example.query for example in eval_dataset.examples]
eval_answers = [example.reference_answer for example in eval_dataset.examples]

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533588728, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="G1gDWlUxjooJ"
len(eval_questions)

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533588728, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="_HeCZJ1xYKYH"
eval_questions[0]

# %% executionInfo={"elapsed": 6, "status": "aborted", "timestamp": 1703533588728, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="47guGTdTYOEi"
eval_answers[0]

# %% [markdown] id="tCNyxGNYgaxl"
# To be consistent we will fix evaluation query

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533588729, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="AABMc2Uxgew_"
eval_query = eval_questions[0]

# %% [markdown]
# <b> Check https://openai.com/pricing to select the less costlier variant of an LLM.<b>

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533588729, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="2XFysSdSX7pS"
# Fix gpt-4o-mini LLM for generating response
gpt4o_mini = OpenAI(temperature=0, model="gpt-4o-mini")

# Fix GPT-4o LLM for evaluation
gpt4 = OpenAI(temperature=0, model="gpt-4o")

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533588729, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="DWLP0Rk8Yj5O"
# create vector index
vector_index = VectorStoreIndex.from_documents(
    documents, llm=OpenAI(temperature=0, model="gpt-4o")
)

# Query engine to generate response
query_engine = vector_index.as_query_engine()

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533588729, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="fzOT-SiFsABn"
retriever = vector_index.as_retriever(similarity_top_k=3)
nodes = retriever.retrieve(eval_query)

# %% executionInfo={"elapsed": 6, "status": "aborted", "timestamp": 1703533588729, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="lB9nry-UsMeR"
from IPython.display import display, HTML
display(HTML(f'<p style="font-size:14px">{nodes[1].get_text()}</p>'))

# %% [markdown] id="8gs6eBCIX2yM"
# ## Context Relevency Evaluation
#
# Measures if the response + source nodes match the query.

# %% executionInfo={"elapsed": 6, "status": "aborted", "timestamp": 1703533588729, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="jfTwgJ5SXoeB"
# Create RelevancyEvaluator using GPT-4 LLM
relevancy_evaluator = RelevancyEvaluator(llm=gpt4)

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533588730, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="6Bw9KWY-YflD"
# Generate response
response_vector = query_engine.query(eval_query)

# %%
response_vector.response

# %%
print(response_vector.get_formatted_sources())

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533588730, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="6Bw9KWY-YflD"
# Evaluation
eval_result = relevancy_evaluator.evaluate_response(query=eval_query, response=response_vector)

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533588730, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="3JMGtE-JaQWd"
eval_result.query

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533588730, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="3irb7WZ-cW84"
eval_result.response

# %% executionInfo={"elapsed": 6, "status": "aborted", "timestamp": 1703533588730, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="GBFxnKgXcZlT"
eval_result.passing

# %% [markdown] id="qaDtmBEjhGQ8"
# Relevancy evaluation with multiple source nodes.

# %% executionInfo={"elapsed": 313968, "status": "aborted", "timestamp": 1703533588731, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="QOhxXsMIgVH8"
# Create Query Engine with similarity_top_k=3
query_engine = vector_index.as_query_engine(similarity_top_k=3)

# Create response
response_vector = query_engine.query(eval_query)

# Evaluate with each source node
eval_source_result_full = [
    relevancy_evaluator.evaluate(
        query=eval_query,
        response=response_vector.response,
        contexts=[source_node.get_content()],
    )
    for source_node in response_vector.source_nodes
]

# Evaluation result
eval_source_result = [
    "Pass" if result.passing else "Fail" for result in eval_source_result_full
]

# %% executionInfo={"elapsed": 313966, "status": "aborted", "timestamp": 1703533588731, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="-euW1W3VhgPx"
eval_source_result

# %% [markdown] id="Rrd_7kufgozj"
# ## Faithfullness Evaluator
#
#  Measures if the response from a query engine matches any source nodes. This is useful for measuring if the response was hallucinated.

# %% executionInfo={"elapsed": 313966, "status": "aborted", "timestamp": 1703533588731, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="Pb3d08hrclbA"
faithfulness_evaluator = FaithfulnessEvaluator(llm=gpt4)

# %% executionInfo={"elapsed": 313963, "status": "aborted", "timestamp": 1703533588731, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="hWOKhDVTdm5q"
eval_result = faithfulness_evaluator.evaluate_response(response=response_vector)

# %% executionInfo={"elapsed": 313961, "status": "aborted", "timestamp": 1703533588731, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="yH977tlPk6HQ"
eval_result.feedback

# %% executionInfo={"elapsed": 313960, "status": "aborted", "timestamp": 1703533588731, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="nsiWpNXHdsoz"
eval_result.passing

# %% executionInfo={"elapsed": 313959, "status": "aborted", "timestamp": 1703533588732, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="JLdKEvMefTIO"
eval_result.score

# %% [markdown] id="SqsEWzF1i1Rk"
# ## Correctness Evaluator
#
# Evaluates the relevance and correctness of a generated answer against a reference answer.

# %% executionInfo={"elapsed": 313959, "status": "aborted", "timestamp": 1703533588732, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="ZdX4-K-NfNIh"
correctness_evaluator = CorrectnessEvaluator(llm=gpt4)

# %% executionInfo={"elapsed": 313956, "status": "aborted", "timestamp": 1703533588732, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="6fOKYRacjIkm"
eval_reference_answer = eval_answers[0]

correctness_result = correctness_evaluator.evaluate(
    query=eval_query,
    response=response_vector.response,
    reference=eval_reference_answer,
)

# %% executionInfo={"elapsed": 313955, "status": "aborted", "timestamp": 1703533588732, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="BbWAp_krjSCI"
correctness_result.score

# %% executionInfo={"elapsed": 313953, "status": "aborted", "timestamp": 1703533588732, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="qBk5I1i9jWM_"
correctness_result.passing

# %% executionInfo={"elapsed": 313952, "status": "aborted", "timestamp": 1703533588733, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="66aEQczhjXpO"
correctness_result.feedback

# %% [markdown] id="RcEM9N4HRjHf"
# ## BatchEvalRunner - Run Evaluations in batch manner.

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533589254, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="SFK9CyRGQ5cH"
from llama_index.core.evaluation import BatchEvalRunner

runner = BatchEvalRunner(
    {
     "faithfulness": faithfulness_evaluator,
     "relevancy": relevancy_evaluator,
     "correctness": correctness_evaluator
     },
    workers=8,
)

eval_results = await runner.aevaluate_queries(
    query_engine, queries=eval_questions, reference = eval_answers
)


# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533589254, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="JJ0ObJrmQ97A"
def get_eval_results(key, eval_results):
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    score = correct / len(results)
    print(f"{key} Score: {score}")
    return score


# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533589254, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="i4UxjfvWQ_9e"
_ = get_eval_results("faithfulness", eval_results)

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533589255, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="Qu-6-JV5RDX3"
_ = get_eval_results("relevancy", eval_results)

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533589255, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="IHWJi1S3ZNUm"
_ = get_eval_results("correctness", eval_results)

# %% [markdown] id="DQUFlSGmRMKm"
# ## Benchmark using [LlamaDatasets](https://llamahub.ai/?tab=llama_datasets).
#
# It's a 3 step process:
#
# 1. Download dataset
# 2. Build your RAG Pipeline
# 3. Evaluate using RagEvaluatorPack.

# %%
# !llamaindex-cli download-llamapack RagEvaluatorPack --download-dir ./rag_evaluator_pack

# %%
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.packs.rag_evaluator import RagEvaluatorPack
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core import VectorStoreIndex

# %% [markdown]
# Download the required files from the below link and move them to the folders mentioned below in the code

# %% [markdown]
# https://github.com/run-llama/llama-datasets/tree/main/llama_datasets/paul_graham_essay

# %%
rag_dataset = LabelledRagDataset.from_json("./data/rag_dataset.json")
documents = SimpleDirectoryReader(input_dir="./data/source_files").load_data()

# %%
rag_dataset.to_pandas()

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533589255, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="1CHzjKlKSvkI"
rag_dataset.examples[0] #query, reference_answer

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533589255, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="1vycmNuESpm_"
# build basic RAG system
index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine()

# %%
rag_evaluator_pack = RagEvaluatorPack(rag_dataset=rag_dataset, query_engine=query_engine, judge_llm=gpt4)

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533589255, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="VH_9noT1Srsr"
# evaluate using the RagEvaluatorPack
benchmark_df = await rag_evaluator_pack.arun(
    batch_size=20,  # batches the number of openai api calls to make
    sleep_time_in_seconds=1,  # seconds to sleep before making an api call
)

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533589255, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="LsYYEXI-St__"
benchmark_df

# %% [markdown] id="aX7xg4hAohDl"
# # Retrieval Evaluation
#
# Evaluates the quality of any Retriever module defined in LlamaIndex.
#
# To assess the quality of a Retriever module in LlamaIndex, we use metrics like hit-rate and MRR. These compare retrieved results to ground-truth context for any question. For simpler evaluation dataset creation, we utilize synthetic data generation.

# %% [markdown] id="qXKPpQ0tP76P"
# Hit Rate:
# MRR:
#
# Document -> D
#
# D -> N1, N2, N3, N4, N5 -> Index/ Retriever
#
# (Q1, N1)
# (Q2, N1)
# (Q3, N2)
# (Q4, N2)
# (Q5, N3)
# (Q6, N3)
# (Q7, N4)
# (Q8, N4)
# (Q9, N5)
# (Q10, N5)
#
# Q1 -> Index/ Retriever -> N2, N1, N3 -> 1 -> 1/2
#
# Q2 -> Index/ Retriever -> N5, N4, N3 -> 0 -> 0
#
# Q3 -> Index/ Retriever -> N1, N2, N3 -> 1 -> 1/2
#
# Q4 -> Index/ Retriever -> N2, N3, N5 -> 1 -> 1/1
#
# Q5 -> Index/ Retriever -> N3, N1, N4 -> 1 -> 1/1
#
# Q6 -> Index/ Retriever -> N1, N2, N3 -> 1 -> 1/3
#
# Q7 -> Index/ Retriever -> N4, N1, N2 -> 1 -> 1/1
#
# Q8 -> Index/ Retriever -> N1, N3, N4 -> 1 -> 1/3
#
# Q9 -> Index/ Retriever -> N2, N3, N4 -> 0 -> 0
#
# Q10 -> Index/ Retriever -> N2, N5, N3 -> 1 -> 1/2
#
# Hit Rate: 8/10 -> 80%
#
# MRR: (0.5 + 0 + 0.5 + 1 + 1 + 0.33 + 1 + 0.33 + 0 + 0.5)/10 -> 55%

# %%
from llama_index.core.text_splitter import SentenceSplitter

# %%
from llama_index.embeddings.openai import OpenAIEmbedding

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533589256, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="wFtWgxqj1x7D"
reader = SimpleDirectoryReader("./data/paul_graham/")
documents = reader.load_data()

# create parser and parse document into nodes
parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
nodes = parser(documents)

# %% executionInfo={"elapsed": 7, "status": "aborted", "timestamp": 1703533589256, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="L8WLcpA-12LB"
vector_index = VectorStoreIndex(nodes, embed_model=OpenAIEmbedding(model='text-embedding-3-small'))

# %% executionInfo={"elapsed": 8, "status": "aborted", "timestamp": 1703533589257, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="XZllZ2u5oj0X"
# Define the retriever
retriever = vector_index.as_retriever(similarity_top_k=2)

# %% executionInfo={"elapsed": 9, "status": "aborted", "timestamp": 1703533589258, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="A5R3H3erqx_i"
retrieved_nodes = retriever.retrieve(eval_query)

# %%
# !pip install matplotlib

# %% executionInfo={"elapsed": 9, "status": "aborted", "timestamp": 1703533589258, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="VMHGPbUiqned"
from llama_index.core.response.notebook_utils import display_source_node

for node in retrieved_nodes:
    display_source_node(node, source_length=2000)

# %% executionInfo={"elapsed": 9, "status": "aborted", "timestamp": 1703533589258, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="R2EzD6m_rDU4"
qa_dataset = generate_question_context_pairs(nodes[0:2], llm=gpt4, num_questions_per_chunk=2)

# %%
qa_dataset.queries

# %%
# !pip install llama_index.llms.groq

# %%
from llama_index.llms.groq import Groq

# %%
groq = Groq(model='llama-3.2-90b-vision-preview')

# %%
DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."

Just provide the questions and nothing else

"""

# %% executionInfo={"elapsed": 9, "status": "aborted", "timestamp": 1703533589258, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="R2EzD6m_rDU4"
qa_dataset_groq = generate_question_context_pairs(nodes[0:3], llm=groq, num_questions_per_chunk=2, 
                                                  qa_generate_prompt_tmpl=DEFAULT_QA_GENERATE_PROMPT_TMPL)

# %%
qa_dataset_groq.queries

# %% executionInfo={"elapsed": 9, "status": "aborted", "timestamp": 1703533589258, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="i9sidRY0xNz-"
queries = qa_dataset.queries.values()
print(list(queries)[:1])

# %% executionInfo={"elapsed": 9, "status": "aborted", "timestamp": 1703533589258, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="ecH5iAUrTaAW"
len(list(queries))

# %% executionInfo={"elapsed": 9, "status": "aborted", "timestamp": 1703533589258, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="mm5GCNZoriDu"
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)

# %% executionInfo={"elapsed": 9, "status": "aborted", "timestamp": 1703533589258, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="Yb5p06r0xYOR"
# try it out on a sample query
sample_id, sample_query = list(qa_dataset.queries.items())[0]
sample_expected = qa_dataset.relevant_docs[sample_id]

eval_result = retriever_evaluator.evaluate(sample_query, sample_expected)
print(eval_result)

# %% executionInfo={"elapsed": 9, "status": "aborted", "timestamp": 1703533589258, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="MF4-RWsDrnMJ"
# try it out on an entire dataset
eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)


# %% executionInfo={"elapsed": 10, "status": "aborted", "timestamp": 1703533589259, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="mxiNl6TurpZQ"
def display_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    metric_df = pd.DataFrame(
        {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}
    )

    return metric_df


# %% executionInfo={"elapsed": 9, "status": "aborted", "timestamp": 1703533589259, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="yZeIaMWKrsY1"
display_results("top-2 eval", eval_results)

# %%
