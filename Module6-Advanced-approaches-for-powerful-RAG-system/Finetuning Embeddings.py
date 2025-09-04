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

# %% [markdown] id="28a8b793"
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/finetuning/embeddings/finetune_embedding.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="551753b7-6cd2-4f81-aec0-da119e4705ad"
# # Finetune Embeddings
#
# In this notebook, we show users how to finetune their own embedding models.
#
# We go through three main sections:
# 1. Preparing the data (our `generate_qa_embedding_pairs` function makes this easy)
# 2. Finetuning the model (using our `SentenceTransformersFinetuneEngine`)
# 3. Evaluating the model on a validation knowledge corpus

# %% [markdown]
# <b> If you face any errors in running this notebook, you run the code mentioned in the below link in the google colab <b>

# %% [markdown]
# https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/

# %% [markdown]
# pip install llama-index-finetuning

# %% [markdown] id="99afd542-fc47-44ac-aed0-b3684108dba5"
# ## Generate Corpus
#
# First, we create the corpus of text chunks by leveraging LlamaIndex to load some financial PDFs, and parsing/chunking into plain text chunks.

# %% id="9280d438-b6bd-4ccf-a730-7c8bb3ebdbeb"
import json

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import MetadataMode

# %% [markdown] id="73c42620"
# Download Data

# %% [markdown]
# download the 'uber_10k' and 'lyft_10k' data files from the below links and move them into the folder that contains this notebook

# %% [markdown] id="d8e11b0c" outputId="64027a63-a47a-42b9-8adb-d771a0752f5b"
# https://github.com/run-llama/llama_index/blob/9607a05a923ddf07deee86a56d386b42943ce381/docs/docs/examples/data/10k/uber_2021.pdf
# https://github.com/run-llama/llama_index/blob/9607a05a923ddf07deee86a56d386b42943ce381/docs/docs/examples/data/10k/lyft_2021.pdf

# %% id="c5e890bc-557b-4d3c-bede-3e80dfeeee18"
TRAIN_FILES = ["./lyft_2021.pdf"]
VAL_FILES = ["./uber_2021.pdf"]

TRAIN_CORPUS_FPATH = "./data/train_corpus.json"
VAL_CORPUS_FPATH = "./data/val_corpus.json"


# %% id="1da871c1-9d58-467a-92fd-06ed3d94534b"
def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes


# %% [markdown] id="53056d8b-3b4c-4364-9b07-a375aa84330b"
# We do a very naive train/val split by having the Lyft corpus as the train dataset, and the Uber corpus as the val dataset.

# %% colab={"referenced_widgets": ["15ef5e0c899d48be9a7f6da0c642a798", "564db8b32b274a428edc854878721a72"]} id="d3651c77-d085-4fbc-bb34-61f143ad6674" outputId="71676cc3-2c69-4f71-909d-881ba44c62f7"
train_nodes = load_corpus(TRAIN_FILES, verbose=True)
val_nodes = load_corpus(VAL_FILES, verbose=True)

# %% [markdown] id="b4482c48-844b-448b-9552-3f38b455645c"
# ### Generate synthetic queries
#
# Now, we use an LLM (gpt-3.5-turbo) to generate questions using each text chunk in the corpus as context.
#
# Each pair of (generated question, text chunk used as context) becomes a datapoint in the finetuning dataset (either for training or evaluation).

# %% id="580334ce-ddaa-4cc0-8c3e-7294d11e4d2f"
from llama_index.finetuning import generate_qa_embedding_pairs

# %% id="ef43fe59-a29c-481b-b086-e98e55016d3e" outputId="11691c3c-21a9-4dbd-ce46-5fa1a946a1bd"
train_dataset = generate_qa_embedding_pairs(train_nodes)
val_dataset = generate_qa_embedding_pairs(val_nodes)

# %%
train_dataset.save_json("train_dataset.json")
val_dataset.save_json("val_dataset.json")

# %%
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

# %% id="743f163c-25df-4c18-9abe-05052b034d70"
# [Optional] Load
train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")

# %% id="b60a2e06-d70b-41dc-b1a7-42795e420260" outputId="2766c596-7132-46f7-e50c-a43337c2e86a"
list(train_dataset.queries.values())[1]

# %% [markdown] id="62368cb8-a303-48b1-8429-5e3655abcc3b"
# ## Run Embedding Finetuning

# %% id="c1d08066-5f00-48f1-b12a-e80bc193d4c0"
from llama_index.finetuning import SentenceTransformersFinetuneEngine

# %% colab={"referenced_widgets": ["cac8b16ac8f74fdc9b881b5ffc5c31f9", "3986867ebce44086a636bb543046805d", "fea463ec47974745a0c4721babc7de95", "2130d3f7b58947649b2dc096da3b25d6", "bc87380cd84a42ffa8448acd784a7492", "6cb0e9a054b249a3a1d8e2b572f1539e", "fc2ffca6c32a4bf8a9b9b52096da6cac", "af5c76b2ad90448d9a6b65c122abdcc1", "5aa977961092498c93f6dfdbf8576f8b", "05ea15f90ddd4966a83a3f20e7f26cbe", "638ddd027e8542bb8156ca36d0b48db6", "befc300b48da4fabb7dd7304cdde214a", "8a33b5877cd14fad8d773c61eae60b22"]} id="26625ab5-ddc9-4dbd-9936-39b69c6a7cdc" outputId="9d847aed-8c77-495b-9ed9-b008878eff51"
finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-small-en",
    model_output_path="test_model",
    val_dataset=val_dataset,
)

# %% colab={"referenced_widgets": ["7dbc03b853514f4480fbdb41ab5f1e8c", "9298054a85c542b1855a3ae631114945", "6d0a19bd54d74714ba38acc8cc107aee"]} id="28ad99e6-dd9d-485a-86e9-1845cf51802b" outputId="69e75c82-4b0f-47be-ef86-4855df8bfddb"
finetune_engine.finetune()

# %% id="467a2ba2-e7e6-4025-8887-cac6e7ecb493"
embed_model = finetune_engine.get_finetuned_model()

# %% [markdown] id="828dd6fe-9a8a-419b-8663-56d81ce73774"
# ## Evaluate Finetuned Model

# %% [markdown] id="f4a66b83-4cbb-4374-a632-0f1bb2b785ab"
# In this section, we evaluate 3 different embedding models:
# 1. proprietary OpenAI embedding,
# 2. open source `BAAI/bge-small-en`, and
# 3. our finetuned embedding model.
#
# We evaluate the models using **hit rate** metric
#
# We show that finetuning on synthetic (LLM-generated) dataset significantly improve upon an opensource embedding model.

# %% id="57d5176f-1f21-4bcb-adf5-da1c4cccb8d3"
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from tqdm.notebook import tqdm
import pandas as pd


# %% [markdown] id="dda4c2b8-1ad8-420c-83d2-b88e0519895d"
# ### Define eval function

# %% [markdown] id="398c24d3-3d72-4ce8-94a4-2da9c1b2605c"
# **Option 1**: We use a simple **hit rate** metric for evaluation:
# * for each (query, relevant_doc) pair,
# * we retrieve top-k documents with the query,  and
# * it's a **hit** if the results contain the relevant_doc.
#
# This approach is very simple and intuitive, and we can apply it to both the proprietary OpenAI embedding as well as our open source and fine-tuned embedding models.

# %% id="b89401d3-a157-4f96-86d4-212e631a54bc"
def evaluate(
    dataset,
    embed_model,
    top_k=5,
    verbose=False,
):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(
        nodes, embed_model=embed_model, show_progress=True
    )
    retriever = index.as_retriever(similarity_top_k=top_k)

    eval_results = []
    for query_id, query in tqdm(queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

        eval_result = {
            "is_hit": is_hit,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results.append(eval_result)
    return eval_results


# %% [markdown] id="af2d33dd-c39f-4c05-8adc-65db12163c88"
# ### Run Evals

# %% [markdown] id="c630aa25-2395-4a8b-83cf-2885fbc862f4"
# #### OpenAI
#
# Note: this might take a few minutes to run since we have to embed the corpus and queries

# %% colab={"referenced_widgets": ["3de7bb1fc4d541a2beb27e33dcdb6557", "2d063690aa0e4b8180c0877e0e25e316"]} id="61a0784f-415e-4d3a-8c88-757b28b9e5df" outputId="c58e6956-1595-4997-d728-27c86e78c455"
embed_model = OpenAIEmbedding(model='text-embedding-3-small')
val_results = evaluate(val_dataset, embed_model)

# %% id="ccc73212-fc53-48c1-b347-f5ee3a29ae82"
df_opanai = pd.DataFrame(val_results)

# %% id="25eb61bb-c287-40fe-b3c7-bbfc2d2b1b94" outputId="1f1a88f9-a099-4d8b-8fda-6de0a1c8b1a9"
hit_rate = df_opanai["is_hit"].mean()
hit_rate

# %% [markdown] id="a1bd6c62-65a8-4f72-a67c-d0d62c92d7d1"
# ### BAAI/bge-small-en

# %% colab={"referenced_widgets": ["5e5c6e7b38db4b8c82e57880eba575ed", "bbd446b1e8ed481ab9a4c68a54cd73db", "e6ef760faafd437eaaef13c521c74d64", "2c8f594d7732403981a037f8c6e353b8", "c9481470c3984db3bdac21057a8c0e7c", "64fce335b7634dbe94d571ea1bbcd529", "f2c714bad4ed4d33a546ea8fcac943e2", "355343b297264836b4d7ef50751e5080"]} id="24454aeb-9e3e-4954-ab70-647102ed7f82" outputId="c80a65dd-08b7-452c-851e-975e49b9e636"
bge = "local:BAAI/bge-small-en"
bge_val_results = evaluate(val_dataset, bge)

# %% id="2da27e48-1c90-4994-aac4-96b5b1638647"
df_bge = pd.DataFrame(bge_val_results)

# %% id="3ddc4fe0-b240-4c15-9b2d-a4c79f9aaac2" outputId="bd06efc6-7221-45b5-d218-b45ede5f15d8"
hit_rate_bge = df_bge["is_hit"].mean()
hit_rate_bge

# %% [markdown] id="1fd87550-f547-4b8b-b21a-f72b355e2cd7"
# ### Finetuned

# %% colab={"referenced_widgets": ["41c6a270908f4ef5b8d87f139f8aa5ce", "1f451939dc004748b26a652b0509cbdf"]} id="402dd440-1934-4778-8ff5-28e15cf1f2d3" outputId="58bc1881-06c7-409d-aa2d-ecb9ee603159"
finetuned = "local:test_model"
val_results_finetuned = evaluate(val_dataset, finetuned)

# %% id="ffd24643-17cb-4773-a535-77f3f8fa2d48"
df_finetuned = pd.DataFrame(val_results_finetuned)

# %% id="ec1dccd1-bbd4-427f-a520-b1011643d83b" outputId="b683badb-5c2b-4562-d103-05dbe992c800"
hit_rate_finetuned = df_finetuned["is_hit"].mean()
hit_rate_finetuned

# %% [markdown] id="fbc290bc-5cc3-4ee4-b8ab-e68371441643"
# ### Summary of Results

# %% [markdown] id="6f906a11-6a95-4f10-9069-140bf5a56246"
# #### Hit rate

# %% id="705fbe3c-2843-4bab-bb5c-16027fc5564b"
df_ada["model"] = "ada"
df_bge["model"] = "bge"
df_finetuned["model"] = "fine_tuned"

# %% [markdown] id="bebc363c-cd07-4dab-916e-1618d16d1254"
# We can see that fine-tuning our small open-source embedding model drastically improve its retrieval quality (even approaching the quality of the proprietary OpenAI embedding)!

# %% id="57f38b4b-1b40-42da-a054-ea9593d3e602" outputId="79ff7056-49c3-4da2-ce47-54539a4fa738"
df_all = pd.concat([df_ada, df_bge, df_finetuned])
df_all.groupby("model").mean("is_hit")


# %% id="21a3b1c0-f1f8-4dc8-9787-7a42943e96ab"
def build_nodes(filepath):

    reader = SimpleDirectoryReader(input_files=[filepath])
    docs = reader.load_data()
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(docs, show_progress=True)

    return nodes


# %% id="3f9a7cdb-1756-4971-99d4-19b5044499c5"
def build_index(nodes, embed_model):
    index = VectorStoreIndex(
        nodes, embed_model=embed_model, show_progress=True
    )
    return index


# %% colab={"referenced_widgets": ["ea1dea0aa9c94de48f33f28700db0bef"]} id="32deba01-3d0c-440b-9c80-afca9c1fce5b" outputId="e153a584-a9f4-48b0-8a64-4c04b8a32e76"
nodes = build_nodes("./data/10k/uber_2021.pdf")

# %% colab={"referenced_widgets": ["a7a3751419f34ea88c901c369c38b389", "1b81dd69f2654cf6b6069275bf1e3afa", "d3d30838722e4f93a6b91ce23278c92e"]} id="e6547828-173a-4036-8718-3fd6fe2aa766" outputId="d23a7331-21ce-4baa-f5d6-c1092782c0c5"
finetuned_embed_index = build_index(nodes, "local:test_model")
base_embed_index = build_index(nodes, "local:BAAI/bge-small-en")
openai_embed_index = build_index(nodes, ada)

# %% id="3452ad9d-2610-46f5-a377-d367bc020cdd"
finetuned_embed_qe = finetuned_embed_index.as_query_engine(similarity_top_k=2)
base_embed_qe = base_embed_index.as_query_engine(similarity_top_k=2)
openai_embed_qe = openai_embed_index.as_query_engine(similarity_top_k=2)

# %% id="015676fa-5868-4645-85c3-9a90598872de"
query = "what is the revenue of uber?"
response1 = finetuned_embed_qe.query(query)
response2 = base_embed_qe.query(query)
response3 = openai_embed_qe.query(query)

# %% id="98dea63c-1b6a-4745-9204-c07ecfc3af75"
print(response1)

# %% id="be9a56a9-73e1-4cb9-8170-4a7f49956c08"
print(response2)

# %% id="350b5fc5-324b-4750-8f81-916f970a2de3"
print(response3)

# %% id="a253e93e-cb43-4a95-b48c-7e0dfda32e2c"
for node in response1.source_nodes:
    print("NODE")
    print(node.get_text())
    print("-----")

# %% id="06821d0e-871f-4bb8-a748-6ade954fa839"
for node in response2.source_nodes:
    print("NODE")
    print(node.get_text())
    print("-----")

# %% id="b5868fa7-23a5-4d62-9964-f0d2bb324b88"
for node in response3.source_nodes:
    print("NODE")
    print(node.get_text())
    print("-----")

# %% id="21b1afe5-cbae-4322-a659-47e5ff240cb3"
