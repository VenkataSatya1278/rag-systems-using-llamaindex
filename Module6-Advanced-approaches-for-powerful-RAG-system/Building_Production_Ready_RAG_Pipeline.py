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

# %% [markdown] id="l7Tqx3WUwpRh"
# # Building Production Ready RAG Pipeline
#
# In this notebook you will learn to build a Production ready RAG Pipeline on `Attention is All You Need` paper. We will use `Sentence Window Index` to build a basic RAG pipeline and iterate over different parameters to make it production ready.
#
# Following are the steps involved:
#
# 1. Download Data
# 2. Load Data
# 3. Build Evaluation Dataset.
# 4. Download `RagEvaluatorPack`.
# 5. Define LLM, Embedding Model.
# 6. Build RAG with `Sentence Window` approach.
# 7. Evaluate RAG Pipeline.
# 8. Create functions to build index, evaluate.
# 9. Tune different parameters to improve metrics and make it production ready.

# %% [markdown] id="qwSFUc760Uxr"
# ## Download `Attention is all you need` paper.

# %%
# attach to the same event-loop
import nest_asyncio

nest_asyncio.apply()

# %% colab={"base_uri": "https://localhost:8080/"} id="lFN9ctrmtFFB" outputId="82e732a4-fde2-4d79-b351-5dd2cb351564"
# !mkdir './data'
# !wget --user-agent="Mozilla" "https://arxiv.org/pdf/1706.03762.pdf" -O "./data/attention_is_all_you_need.pdf"

# %% [markdown] id="EAxym48t0ZVw"
# ## Set `OpenAI` keys.

# %%
import os 
from dotenv import load_dotenv, find_dotenv

# %%
load_dotenv('D:/Training/FAA-Training/Beyond-the-Prompt-Practical-RAG-for-Real-World-AI/RAG-systems-using-LlamaIndex/RAG-System-Using-LamaIndex/.env')

# %%
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# %% [markdown] id="F05_iH6f0e58"
# ## Load Data.
#
# We will use first 10 pages and skip paper references in the paper.

# %%
# !pip install spacy

# %% id="1RNR-LJBtvwE"
from llama_index.core import SimpleDirectoryReader

data = SimpleDirectoryReader('./data/').load_data()

documents = data[:1]

# %% [markdown] id="uzCkljUn0xdW"
# ## Generate Evaluation dataset using `RagDatasetGenerator` and `GPT-4`

# %%
from llama_index.llms.openai import OpenAI
from llama_index.core.llama_dataset.generator import RagDatasetGenerator

# %% colab={"base_uri": "https://localhost:8080/", "height": 240, "referenced_widgets": ["754fafb9284449809ac1159bde9bbf1c", "4c36495a9418491c9a77bd0bf1303a75", "1f07df3d7a9148b2827f830fa2dfd5a2", "48664bcefdbf4473b574b6890ce63181", "375f9b080ed040029ad4891c510e9129", "9579b32a07754fb487106c4577ba007f", "5fd4f145f21c4003b0c09012c4cac409", "b5d2a2d7199d47af9860baf26177d123", "19dc14d86ca5428b9c1f88973a89e178", "8086b1d033f14a61ab61e78fe127b4fc", "61c114c7e2b3423699c871382b230e13"]} id="9Qig7neff4WK" outputId="223dd714-bf6c-4810-a823-71bab8fcd6a5"
gpt4 = OpenAI(model='gpt-4-0125-preview', temperature=0.1)

dataset_generator = RagDatasetGenerator.from_documents(
    documents,
    llm=gpt4,
    num_questions_per_chunk=2,
    show_progress=True,
)

eval_dataset = dataset_generator.generate_dataset_from_nodes()

# %% [markdown] id="QY1gmjyJ1H0S"
# ## Import `RagEvaluatorPack` for evaluation.

# %% id="diFoF7ud1IVq"
from llama_index.packs.rag_evaluator import RagEvaluatorPack

# %% [markdown] id="FC6tkEOv1OlP"
# ## Define LLM

# %% id="8gcpCJdGfZ4B"
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

# %% [markdown] id="9HCoJGhL1Psf"
# ## Define Embedding Model

# %% id="sNJBYz651PLY"
#embed_model = "local:BAAI/bge-small-en-v1.5"

# %% [markdown] id="Nc6KGdMq1ZTh"
# ## Build RAG pipeline with `SentenceWindow`

# %% id="TSOJsiKluOSv"
from llama_index.core.node_parser import SentenceWindowNodeParser

# create the sentence window node parser w/ default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=1,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# %% id="Q5O30K6Wu5wv"
from llama_index.core import Document

document = Document(text="\n\n".join([doc.text for doc in documents]))

# %%
# !pip install llama_index.embeddings.openai

# %%
from llama_index.embeddings.openai import OpenAIEmbedding
embed_model = OpenAIEmbedding(model='text-embedding-3-small', api_key=OPENAI_API_KEY)

# %% id="2S-5gXAovJdF"
from llama_index.core import VectorStoreIndex

sentence_index = VectorStoreIndex.from_documents(
    [document], embed_model=embed_model, llm=llm)

# %% colab={"base_uri": "https://localhost:8080/", "height": 209, "referenced_widgets": ["813473b1bf214f5caa7405e7527ec63f", "1c12da8521714089a542ff1ecdf2e78a", "f90c16ea546b4dae8cb7471451614d89", "877877be05714842ae30f566087f71ea", "731cd7f4b4794a3f95583cbf17c9b09e", "f6708109ac534ee39d0fab8872a9154d", "4cfa8d1a2f294bd0b390174a228b4227", "2d75faeef7f84898aaafb7aa60151466", "f7d31179ed0440f5bd04dbbf8f5486c1", "b1f4c714d9be429689afa463a63cffac", "9d279821e34e40be8a1cc6bf2a34c0dc", "5e2092f8646c4beb823d0322149554fb", "482ce14b17da4c0eb77b173e3674ea93", "a4060264848342c2a3cb35d21f71cc88", "5c9fe87de41442adbeeff8e2df9dd7f0", "b5d8ee53fc0b45b2af1f2552e91d8f4d", "4873883eaa134bce98c339d47b3e6c3d", "bf10302546914fb2bbc4e308a6e87199", "1ff7898523d64458a3a3605672a305c4", "bef786b8c5604fb0b1a09b6800a75524", "8166c17b7833460482d630184d522af2", "50c424b554be412c91ce008b8b200ea2", "592c6d3572764c41bae49056f3b3fff5", "97d7c8f76f864ae6ab62744e2e5174ae", "edb019aac0ed441fbbf6103499683eb3", "218ef6ad97b84d86b6d60daac9ac9f2d", "f7df97be499b4eb58b0f797b4d9652bd", "cedb356395164cfc8246b4567e8ddda9", "bad3c0a839384b6294f13afb109f253b", "899101431de644cfa7e1dfe57085bc10", "e95b765cf7d24cfaa24f55688f8ee27b", "f80dac7fd2934720b346238867a7b737", "15348ef375c9491a86dc422ac50a818e", "4d95292222024f80b17d9f55d162f0e0", "5d305a4a16d247e6bab0b55e549da118", "5ca6c7c1a59144f0b0d756e0ad3ee9b7", "f7dbd9f176fc4571b863745d737ade24", "ba5f4e0b33524a5d971d7f300e6033dc", "ced1be9ac12b4d0bb57fbd5735e79f24", "73d83bd16b424759aab7514dea907333", "a43a76fa54164c3ebc4739ca08f3a258", "48821dcb13264ba1b7a3678dc019da44", "c863807bac2c4bf7b9972197380e6586", "d57a6e29eb60410c9b9b311c1aea5d68", "8ea70f1d75fb4d62a6af7c0cde5bbb6a", "3c72bf1aa11840478e3bb2b6d747540d", "5f7599bbff464fb186c9c9c3f3757fc7", "aedbf2850e514ec6b5eb65b7323559db", "76a93faddc4c4a9fb2a70e37e184622b", "d29f77e655d9469796b74fa26f776064", "92351ac6e62745aaae46f7d9b8f6e09a", "f59499eb406d45eab887e7948a1de0a5", "4d515aabb2b24d0585b6ddea0fd3dccc", "32306a767e734fe19a4a490d2d2d74c4", "08fabc01712543b2ab99c9f8469870ef", "98b409d6e6d84068a02148097a453ee2", "4b1fa806c1e04b62893f9272eed8d432", "d48b55b6e7ad4cda84292124d66c7517", "1a02cb99e3a74f228460dd1ba5760c5d", "e3d526d479f24e28a5317864ceb4f257", "a9eca53d2691447083a4baeef2e8b3e8", "1aec0e554c8e4778ad5c3318eb07a35a", "87c3f92590ce48f3a4086ab57d162c82", "d10a4bb14d2f440b8475a8736e88a536", "d870ef8d19ba4779a5063a8de90583fa", "f02c0bbcefc844a69c63036168dedab4"]} id="2j4RdycNvW_A" outputId="69c731d8-edbc-4333-bcfb-d6d8725d52b0"
from llama_index.core.indices.postprocessor import SentenceTransformerRerank

rerank = SentenceTransformerRerank(
    top_n=2, model="BAAI/bge-reranker-base"
)

# %% id="VAXKlbm0vtaa"
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor

postproc = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)

# %% id="EJxLIsiDviRG"
query_engine = sentence_index.as_query_engine(
    similarity_top_k=2, node_postprocessors=[postproc, rerank]
)

# %% colab={"base_uri": "https://localhost:8080/"} id="UUeRB0_IvwsU" outputId="7fabdf0e-83ab-43fa-da2b-f7b8afe9011e"
response = query_engine.query('is the paper from google research?')
print(response)

# %% [markdown] id="EK64Wge51iY0"
# ## Evaluate RAG pipeline

# %% colab={"base_uri": "https://localhost:8080/"} id="zlbZjNmsg30-" outputId="746d2755-d023-4383-e8f1-839d3817be10"
rag_evaluator_pack = RagEvaluatorPack(
    rag_dataset=eval_dataset,
    query_engine=query_engine,
    judge_llm=gpt4,
)

base_benchmark = await rag_evaluator_pack.arun(
    batch_size=10,  # batches the number of openai api calls to make
    sleep_time_in_seconds=1,  # seconds to sleep before making an api call
)


# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="gov3-2YPig_O" outputId="29941616-389d-441e-8af1-34232eef7d9c"
base_benchmark


# %% [markdown] id="JHHI7YbX1s-r"
# ## Create Functions to build RAG pipeline and Evaluation.
#
# This will make the process of iterating easier for evaluation.

# %% id="fRvPx_LSyOc3"
def build_index(documents, llm=OpenAI(model='gpt-3.5-turbo', temperature=0.1),
                embed_model="local:BAAI/bge-small-en-v1.5", sentence_window_size=3):
    
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    
    sentence_index = VectorStoreIndex.from_documents(
        documents, embed_model=embed_model, llm=llm
    )

    return sentence_index


def setup_query_engine(sentence_index, similarity_top_k=2, rerank_top_n=2, is_rerank = False):
    
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    if is_rerank:
      rerank = SentenceTransformerRerank(
          top_n=rerank_top_n, model="BAAI/bge-reranker-base"
      )
      query_engine = sentence_index.as_query_engine(
          similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
      )
    else:
      query_engine = sentence_index.as_query_engine(
          similarity_top_k=similarity_top_k, node_postprocessors=[postproc]
      )
    return query_engine

async def evaluate(query_engine):
  rag_evaluator_pack = RagEvaluatorPack(
      rag_dataset=eval_dataset,
      query_engine=query_engine
  )

  benchmark_df = await rag_evaluator_pack.arun(
      batch_size=10,  # batches the number of openai api calls to make
      sleep_time_in_seconds=1,  # seconds to sleep before making an api call
  )

  return benchmark_df

def build_index_query_engine(window_size,
                             similarity_top_k,
                             rerank_similarity_top_k,
                             rerank_top_k):
  sentence_index = build_index(
      [document],
      sentence_window_size=window_size,
  )

  query_engine = setup_query_engine(sentence_index,
                                      similarity_top_k=similarity_top_k)

  query_engine_rerank = setup_query_engine(sentence_index,
                                            similarity_top_k=rerank_similarity_top_k,
                                            rerank_top_n=rerank_top_k,
                                            is_rerank=True)
  return sentence_index, query_engine, query_engine_rerank


# %% id="I8agrz3tnJzf"
index, query_engine, query_engine_rerank = build_index_query_engine(1, 2, 4, 2)

# %% colab={"base_uri": "https://localhost:8080/"} id="eRlCvQGckzDE" outputId="6ef0c12f-03cf-46c8-a1b3-27423cd34fdd"
base_benchmark = await evaluate(query_engine)

# %% colab={"base_uri": "https://localhost:8080/"} id="C6HVo6C8ll9Z" outputId="1276cb51-1042-4489-a8af-0885e778fd60"
base_benchmark_rerank = await evaluate(query_engine_rerank)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="TGABLc72mDXu" outputId="4c929dc4-08b2-4581-ab02-6285766dabe2"
base_benchmark

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="0u1htI4RmE4O" outputId="c733a3ad-ef68-444a-9eba-8c4cc7ba6439"
base_benchmark_rerank

# %% [markdown] id="LuDE4VFB16Ex"
# From the metrics we can observe that `correctness` metric is lower (maximum 5) and including `reranker` improved metrics though it decreased the `correctness` metric.
#
# Interesting to see there are no hallucinations as `faithfulness` metric is 1.0

# %% [markdown] id="Vcl5LZdd2VFJ"
# ## Tune parameters to make it production ready.
#
# Let's aim to get `correctness` score of `4.5` and `relevancy` score of more than `0.9`.

# %% [markdown] id="dsg4lTHx2rhJ"
# ### Experiment 1:
#
# Let's increase window size and see if we can improve correctness as it gives more surrounding context.

# %% id="Swo3EFw8mF81"
index, query_engine, query_engine_rerank = build_index_query_engine(3, 2, 4, 2)

# %% colab={"base_uri": "https://localhost:8080/"} id="R5QaQ4Zmng5r" outputId="f15a7e7c-55ed-49b0-97d9-82a94c686895"
benchmark = await evaluate(query_engine)

# %% colab={"base_uri": "https://localhost:8080/"} id="B64gmUUPnkew" outputId="b75116e1-361c-4843-cec2-1f792eefa14b"
benchmark_rerank = await evaluate(query_engine_rerank)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="wmaYqv-JnnYR" outputId="9a386458-5be9-4078-8b7b-955d8165f5ab"
benchmark

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="GvHSxItVnoq2" outputId="f99547d2-99cb-40cd-a0eb-d256e9877d0e"
benchmark_rerank

# %% [markdown] id="nDrX-Ns32628"
# The metrics did improve.

# %% [markdown] id="0hxfpyTK3ARn"
# ### Experiment 2:
#
# Let'r increase `similarity_top_k` and `reranker_top_n` values and see if getting more relevant contexts improves the result.

# %% id="x-V07ZzLnuzX"
index, query_engine, query_engine_rerank = build_index_query_engine(3, 4, 8, 4)

# %% colab={"base_uri": "https://localhost:8080/"} id="SG_V9UdapHql" outputId="263fbc92-838e-4e75-8b58-bdd489f92deb"
benchmark = await evaluate(query_engine)

# %% colab={"base_uri": "https://localhost:8080/"} id="gyvT_GD3pJl6" outputId="78341837-0ee4-4652-ce1e-2e34e7b38a9a"
benchmark_rerank = await evaluate(query_engine_rerank)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="RZludvnjpK-_" outputId="0825745d-806b-4415-c3e8-a6b0dc7878cb"
benchmark

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="fEj439k4pMdR" outputId="82a82141-5693-42f0-d151-c070fc32a222"
benchmark_rerank

# %% [markdown] id="eQBejNuJ3U8b"
# We are close to our goal. We reached a `correctness` score of `4.275` and `relevancy` score of 0.9. The `context similarity score` also improved

# %% [markdown] id="XhH2yU_V3mME"
# ### Experiment 3:
#
# Let's now increase window size and see if it improves the metrics.

# %% id="0E25woCopNSH"
index, query_engine, query_engine_rerank = build_index_query_engine(5, 4, 8, 4)

# %% colab={"base_uri": "https://localhost:8080/"} id="iJJ081vSrJ1c" outputId="54209a45-75aa-4ecb-87a6-c4dfaa041782"
benchmark = await evaluate(query_engine)

# %% colab={"base_uri": "https://localhost:8080/"} id="Uzr-mh8zrMBq" outputId="21c82c97-2153-43eb-d953-90b29c62417f"
benchmark_rerank = await evaluate(query_engine_rerank)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="fGN5qWCDrNbu" outputId="23667d82-a411-4f88-b805-08f4d9cdb3b0"
benchmark

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="6PeHmwPSrQT1" outputId="2f7b074e-05e1-457e-e87b-5d29f4265a63"
benchmark_rerank

# %% [markdown] id="y6s1HecX3x9u"
# We have reached a `relevency` score of `1.0`.

# %% [markdown] id="HVmbBWGA369V"
# ### Experiment 4:
#
# Let'r increase `similarity_top_k` and `reranker_top_n` values and see if getting more relevant contexts improves the result.

# %% id="VvOJwGq-rRD7"
index, query_engine, query_engine_rerank = build_index_query_engine(5, 6, 12, 6)

# %% colab={"base_uri": "https://localhost:8080/"} id="p9AWnxaatpnr" outputId="b033c012-2c57-4d76-e397-d2359271aa50"
benchmark_rerank = await evaluate(query_engine_rerank)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="jMpdHdQJtsgj" outputId="b777d5a1-232b-4003-c1a9-c16d1a62b6bc"
benchmark_rerank

# %% [markdown] id="fyLTwbj74EQD"
# We have reached our goal of `correctness` score of `4.5` and `relevancy` score of `1.0` (>0.9).

# %% [markdown] id="C_cGXTsw4o31"
# ## Observation:
#
# In this project, we looked into building RAG Pipeline, evaluation dataset and tuning different parameters to make it production ready. It should be observed that `reranker` improved metrics in most of the experiments.
#
# Please do remember that we have various other metrics like `chunk_size`, `chunk_overlap`, `embedding model`, `LLM` to experiment.

# %% id="8MtlTYp9ttVA"
