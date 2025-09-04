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

# %% [markdown] id="b07531d9-7473-480d-bee6-c1ee4cbc207c"
# # Automatic Metadata Extraction
#
# In this tutorial, we show you how to perform automated metadata extraction for better retrieval results.
# We use two extractors: a QuestionAnsweredExtractor which generates question/answer pairs from a piece of text, and also a SummaryExtractor which extracts summaries, not only within the current text, but also within adjacent texts.
#
#

# %% [markdown] id="9a4873de-eaa9-4854-8aeb-050704bd894f"
# ## Setup

# %% id="40d399c4-c93c-41bf-9a47-48aefabb75e3"
import nest_asyncio
nest_asyncio.apply()

# %%
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv('D:/Training/FAA-Training/Beyond-the-Prompt-Practical-RAG-for-Real-World-AI/RAG-systems-using-LlamaIndex/RAG-System-Using-LamaIndex/.env')
# load_dotenv('D:/.env')
# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# %%

# %% [markdown] id="7a6ef3fc-0d04-43a2-b0d6-4d8f3c90ef3d"
# ## Define Metadata Extractors
#
# Here we define metadata extractors. We define two metadata extractors:
# - QuestionsAnsweredExtractor
# - SummaryExtractor

# %% id="0adb8e4a-6728-4073-8256-8b3be4ab1e64"
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import MetadataMode

# %% id="a0231dff-7443-46bf-9b9d-759198d3408e"
llm = OpenAI(temperature=0.1, model="gpt-4o", max_tokens=512)  #"gpt-3.5-turbo"

# %%

# %% [markdown] id="2db2cf90-f295-4a3d-a47c-4b2b1dd2d7c5"
# We also show how to instantiate the `SummaryExtractor` and `QuestionsAnsweredExtractor`.

# %% id="3bda151d-6fb8-427e-82fc-0f3bb469d705"
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import SummaryExtractor, QuestionsAnsweredExtractor

# Node parser
node_parser = TokenTextSplitter(separator=" ", chunk_size=256, chunk_overlap=128)

# Question Answer Extractor
question_answer_extractor = QuestionsAnsweredExtractor(questions=3, llm=llm, metadata_mode=MetadataMode.EMBED)

# Summary Extractor
summary_extractor = SummaryExtractor(summaries=["prev", "self", "next"], llm=llm)

# %%

# %% [markdown] id="e4e54937-e9e7-48ed-8600-72cd2f3c529b"
# ## Load in Data, Run Extractors
#
# We load in Eugene's essay (https://eugeneyan.com/writing/llm-patterns/) using our LlamaHub SimpleWebPageReader.
#
# We then run our extractors.

# %%
from llama_index.readers.web import UnstructuredURLLoader

# %%
loader = UnstructuredURLLoader(urls=["https://eugeneyan.com/writing/llm-patterns/"])

# %%
documents = loader.load_data()

# %%
len(documents)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1703425927461, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="f2701c7e-b67e-4c24-98df-73f96e3756a2" outputId="4e3e3a75-b209-4c92-be55-cf437fb30918"
print(documents[0].get_content())

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 696, "status": "ok", "timestamp": 1703426006180, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="g_tDV0BvGdnF" outputId="0fce83d2-82f4-4c1b-9bc2-80af0cff97c2"
orig_nodes = node_parser.get_nodes_from_documents(documents)

# %%
len(orig_nodes)

# %% id="d63b7df2-0e5a-4e98-85ea-88ddcf37c99e"
# take just these 8 nodes for testing
nodes = orig_nodes[20:28]

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 622, "status": "ok", "timestamp": 1703426019697, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="eb2a96d9-03fe-4d60-9829-4fa80f7ff571" outputId="9f1799c8-11e1-40f3-cfbc-4d59e749de54"
print(nodes[3].get_content(metadata_mode="all"))

# %%

# %%

# %% [markdown] id="970f4dd0-d5e2-4bef-abc1-494020a9a2b5"
# ### Run metadata extractors

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10266, "status": "ok", "timestamp": 1703426039671, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="c4f86b9c-4c9d-48e2-915a-328d5887a1a3" outputId="5e75b4fd-2045-476d-d4af-a24ce215d1ad"
# process nodes with metadata extractors
nodes_1 = summary_extractor(nodes)

# %%
nodes_1[3].to_dict()

# %%
nodes_1[3].metadata

# %%
print(nodes_1[3].text)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10266, "status": "ok", "timestamp": 1703426039671, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="c4f86b9c-4c9d-48e2-915a-328d5887a1a3" outputId="5e75b4fd-2045-476d-d4af-a24ce215d1ad"
nodes_1 = question_answer_extractor(nodes_1)

# %%
nodes_1[3].metadata

# %%
print(nodes_1[3].metadata["questions_this_excerpt_can_answer"])

# %%

# %%

# %% [markdown] id="d1d52daa-b57a-4f9a-9f67-70cddd5304a4"
# ### Visualize some sample data

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1703426042563, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="7822c90d-d048-41d1-b64e-8fbcd8f6e65b" outputId="7d77e5fe-a55a-456e-8b56-451d6535099f"
print(nodes_1[3].get_content(metadata_mode="all"))

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1703426059530, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="bbf5ee6a-daf8-4862-a80a-15c03f0db4b0" outputId="0aa5f2c2-8582-4687-f00b-19052e4804bd"
print(nodes_1[1].get_content(metadata_mode="all"))

# %%

# %%

# %% [markdown] id="641ae280-68e0-4003-b5de-d9f7ecdc6423"
# ## Setup RAG Query Engine.
#

# %% id="6e430f74-dd95-4aef-acc1-07aa1f3cbd7f"
from llama_index.core import VectorStoreIndex
from llama_index.core.response.notebook_utils import display_source_node, display_response

# %%
orig_nodes[0].metadata

# %% id="bd729fb8-1e00-4cd0-9505-a86a7daa89d0"
index = VectorStoreIndex(orig_nodes[:20] + nodes_1 + orig_nodes[28:])

# %% id="061b1de8-628f-4dbe-852b-3c467ae86aba"
query_engine = index.as_query_engine()

# %%

# %% [markdown] id="79e264bb-ad46-4ed3-a461-6258ce80f944"
# ### Querying

# %% id="1e1e448d-632c-42a0-ad60-4a315491945f"
query_str = (
    "Can you describe metrics for evaluating text generation quality, compare"
    " them, and tell me about their downsides"
)

response = query_engine.query(query_str)
display_response(response, source_length=1000)

# %%

# %%
