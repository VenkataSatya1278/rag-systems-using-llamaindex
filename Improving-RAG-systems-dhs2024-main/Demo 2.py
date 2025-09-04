# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/dipanjanS/improving-RAG-systems-dhs2024/blob/main/Demo_2_Solutions_for_Missed_Top_Ranked%2C_Not_in_Context%2C_Not_Extracted_%26_Incorrect_Specificity.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="b8kkhS-UgJAR"
# # Solutions for Missed Top Ranked, Not in Context, Not Extracted & Incorrect Specificity
#
#
#
# Here we will explore the following strategies
#
# - Effect of Embedder Models
# - Advanced Retrieval Strategies
# - Chained Retrieval with Rerankers
# - Context Compression Strategies

# %% [markdown] id="L1KvMtf54l0d"
# #### Install OpenAI, HuggingFace and LangChain dependencies

# %% id="2evPp14fy258" colab={"base_uri": "https://localhost:8080/"} outputId="b815ebd4-f09c-4f7a-eef0-a9f47858685e"
# !pip install langchain
# !pip install langchain-openai
# !pip install langchain-community
# !pip install langchain-huggingface
# !pip install langchain-chroma
# !pip install rank_bm25

# %% [markdown] id="-E6SF7YdioWZ"
# ### Enter Open AI and HuggingFace API Tokens

# %% colab={"base_uri": "https://localhost:8080/"} id="eeBpx1YEioWg" outputId="fa0da83e-8037-4f66-aafd-c0a7e5e6a0ee"
from getpass import getpass

OPENAI_KEY = getpass('Enter Open AI API Key: ')

# %% id="h7vnQZC6_eiy" colab={"base_uri": "https://localhost:8080/"} outputId="60ce8e4a-8523-4953-cfb0-b48db66a2b82"
from getpass import getpass

HUGGINGFACEHUB_API_TOKEN = getpass('Enter HuggingFace Auth Token Key: ')

# %% id="x1YSuHNF_lbh"
import os

os.environ['OPENAI_API_KEY'] = OPENAI_KEY
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

# %% [markdown] id="T5y3ahKK5XLx"
# ### Get Wikipedia Data

# %% colab={"base_uri": "https://localhost:8080/"} id="TCtrXjMMj_Lv" outputId="022f0a3c-c1e4-4a32-ce8d-3bf2129913ae"
# !gdown 1oWBnoxBZ1Mpeond8XDUSO6J9oAjcRDyW

# %% id="piqK8E3dj_5I"
import gzip
import json
from langchain.docstore.document import Document

wikipedia_filepath = 'simplewiki-2020-11-01.jsonl.gz'

docs = []
with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        data = json.loads(line.strip())
        #Only add the first paragraph
        docs.append({
                        'metadata': {
                                        'title': data.get('title'),
                                        'article_id': data.get('id')
                        },
                        'data': data.get('paragraphs')[0] # restrict data to first 3 paragraphs to run later modules faster
        })

# %% colab={"base_uri": "https://localhost:8080/"} id="yJy7WIMwlPRt" outputId="1be36af6-25bb-4c84-b145-182d1c0e4a15"
docs = [doc for doc in docs for x in ['india']
              if x in doc['data'].lower().split()]
docs = [Document(page_content=doc['data'],
                 metadata=doc['metadata']) for doc in docs]
len(docs)

# %% colab={"base_uri": "https://localhost:8080/"} id="p4GcRlPclb9c" outputId="0770c5a5-4bd8-46c7-efe8-9ff61ca66f10"
docs[:3]

# %% [markdown] id="fshdD1t95aWt"
# # Exploring Embedding Models

# %% id="u0PlmZr7ik86"
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# check out model details here: https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1
model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

bert_embedder = HuggingFaceEmbeddings(model_name=model_name)

# %% id="Lf9WdOgJkeBV"
from langchain_chroma import Chroma

# create vector DB of docs and embeddings - takes < 30s on Colab
chroma_db1 = Chroma.from_documents(documents=docs,
                                  collection_name='wikipedia_db1',
                                  embedding=bert_embedder,
                                  collection_metadata={"hnsw:space": "cosine"})

retriever1 = chroma_db1.as_retriever(search_type="similarity",
                                     search_kwargs={"k": 2})

# %% id="4czRBjWLe8JH" colab={"base_uri": "https://localhost:8080/"} outputId="98d99cdf-3786-451e-ae4f-43dbb3880238"
query = "what is the capital of India?"
top_docs = retriever1.invoke(query)
top_docs

# %% colab={"base_uri": "https://localhost:8080/"} id="dHZ5pge-l1Bm" outputId="a426ab65-6845-4cc5-9f34-3fc151d544b9"
query = "what is the old capital of India?"
top_docs = retriever1.invoke(query)
top_docs

# %% id="28n79jM-AsBx" colab={"base_uri": "https://localhost:8080/", "height": 369, "referenced_widgets": ["f05ee502b1674804affc242dd845ef4d", "36ea3e95e7de405e90e3a573d2fc3627", "9ff59f4cd931459db81798c9ee4ce3c7", "ec50e3d682a94950aab487091e938ba6", "7d77ab5af9b441c593a1407de4b13ef9", "04bde738dcda40eab85fb63d3e968d21", "c84c62c3ce7c43649321edbb19c8eff0", "5bd4e6bb4dab47799f66a1f0524b30fb", "d8557ce3e62747a68d8a48897aa344e4", "c832bad16f22449890cfe3d84134e51b", "d7128cdbb66c4094a4928bef94c185d6", "4e8a9b49f5ef4eb3ae1864a16581046f", "fcb096f4a3cc4df58d0070369fd4410b", "6dfaa4d9f69841a7a2b60c0730cde0f8", "429bafdf29034374b773b5a40012c709", "737a8c828bf047f2b31855c2d81fa393", "9dd5e13a314c46c9a90e35f853085ca4", "6d57fd33b859493793102394540f64b0", "8f25f3632c634d96923e912e6b39426a", "896d0228ba85473a94508dfc80a97e88", "3cb8639f17d04f369653057a41553026", "4b9219fc58044e33995199c3f81bb118", "872d57b8fdc7472f97c144db4f4af9a7", "150c316b610444119ba4995d7aad30ac", "90bc7d22952d47bfa5086d691a565e58", "8f67adb350404a638b8e5ecb80026460", "085b6d44596d4968838a7c78973b906a", "49d10bdcca0a437e87c56e95062c1fe0", "285f6e434c9a486182d24805d267e831", "d62e87abae0d45e98b95def04656cd08", "d5fc0e1391db404a8c21c7ba26afb1f9", "2c3830f52a4049ba8c7c2dbd1fa263f3", "af4b3e5b22d649f9b48e8a169a8c62b8", "b938e32678a34a20842defd11f93dbbc", "f7c74c1e59d04e3b9b4ffb62cafbb7d3", "d9a7a9c07c794340be90d7ddbba1209b", "bcd6b78629f149f4a0abc9c5e5f75ecc", "167ac0f03c6c47029125d7a8b5979ea6", "2cb769783da441058b96d50de3291cfd", "a5cb2e5858e743b2befc708b5a76e8c0", "35c3e24e85704a4fb0523902d5c47c98", "ec8b24fa765743c4a90b10ac0172e22f", "e36a1fd57b42408ca38a1fc95ec22f9f", "daacd3fd6f8848a0a7550f2c23303747", "9b8d7fec253744e0b75ad71286762307", "f24f3874fee74736ae00ca18107e74c4", "d793aec02b214ec08b11b50811a9a0e6", "8913148ae5e0428fa906ef5d9005fff2", "c49eec656c2244d6a2d005865b0f46ae", "685a0c87796d48159ba36f92eea37678", "0aa75412392e46c08bbee60780361391", "f8456b5ad8b443e2a9272a2c11cf3a85", "da3c2c7f86af4df8898e0f9031903ab4", "8600576c9a494dc39b5eb1226d2c4b38", "df2be16dd3d94f9983abcc7987dd5e12", "33ee885e97ea416cbc6bea61ac4c7128", "7d5407186bda4d7ba987b043af8800ef", "6902c8843fe445c7adda3a83425c8cb9", "d0be77bdcffd444993838c0a677051aa", "2e3e431d03bf4115a731ffc47908e949", "a0dd14d6078f42d9ac28a6f9817e8f5a", "a69c718c8f9948faa3444c5db2543350", "b927682dce7a4d998bfd974693519a51", "9a6e8912ec504df2b750412eb3d6f170", "485b42159d6d4aefb8e4ac6d326acf8b", "c401f0cf2f1d41f58a65c175a76724a5", "f7a92be4aa4643ac99d0bc8d49eb41ab", "d508142821a74995a8df793abc2433db", "888607e580914ff48858a4a67f6bef5e", "482a5b3712ea480c9b2772d02b9cf571", "2decebcbb8294c4f9f9532761b1b4531", "022fcdf81cb14921a5683bb706ae47dd", "1b288dd607ab48b7bed6845e5a12a91d", "47cfec03861344e38300deecc6426c2e", "ad0030f2b96f48d188f83a7f1b47aaaa", "b7963450f7ef4102a71b83f09c559e50", "15970a31b1634559b6e59874c5569993", "55be6808599b445696d9d3ab7b0c91aa", "2f33ec157f724798a04bfe9ffadb83a6", "8f0f93ce80cd4d908f6cb4c42baf409a", "684f336342fd407f898f71cbf875af72", "ff8ee5d1148e4b92877d1a721a1dcd42", "d213c64ee3be43febf824f11105b5d4a", "53b392488d1849d0a1c57b70ad262597", "198e6d3f14244610b537224f7325fc2e", "c480998e77a640ebb67cff80cb2fd1bb", "2f1bb064b88a46ccae0baa75833a4f9c", "f50ad174d5474904b3df0ebbacacc922", "dff06169381f4da1adf5e4f411a4ec0b", "89a1e0df8aad470b968eb6373f4f00b6", "8edb2753a333441299f41f18ae3ae73a", "7e5edd0959e74f78b2a9214c280a10fa", "ff8524f5a77e4c8d8a717d6ef71fc535", "35a3b81d98654f6382528c8192dc0211", "1be2218a391147c6bac72960d451aa7f", "108613fca6ca44dba9fb982a0b925c7f", "59869e8906954a60975338c813b7a9f6", "13cc947499564dab99eb5a6b5e849fe0", "ec87afa2419146c4aa97bd88188d2404", "519a93677a88425bbd9dc977ec825334", "6ad169f2cb20492c8fe9c1af2ecee0ad", "d84816526ee04809a5179969d8ca9ab5", "a4437b4d58f941939cbd0d0be9b48820", "880ab5225f2c4868b1b6bbed2361471b", "784a4ee57b294960ac25839c4a769608", "621a3c2ebecb459aa97503741ab71e25", "60150558bc1b4fed97bad541821a8bcc", "045e814411c5414593c98795d9189c73", "6413f1ccfdc045dca222c59f5fc3c45f", "79bad8778137494b81a8604649d95e8a", "5e4b2cb0f6a54ac9b1c44acb29f9488c", "f0772c5d87a34694a8ff11861b7b1ab9", "429cc8e9b09e461eaacc99fc7f1d56c1", "52f008754d55488f96cefd0fea2435e4", "db76ec8d5ce14550a77ed14e8afb2943", "ac50b90e0a5c4789bba4fb2d48e8ab1c", "b4f9cc79095e444fa997bc15321616bf", "20e9071b78df4609b18317dc88d13ee3", "9abdf3c8d8624eda935cea912deb8950", "162a61a5c822459f85251d20f91f1813", "8320289cb17946eeaa21b1653db03e3c"]} outputId="5e0b0177-f248-4e8f-a89f-27874b667097"
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# check out model details here: https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
model_name = "mixedbread-ai/mxbai-embed-large-v1"

mxbai_embedder = HuggingFaceEmbeddings(model_name=model_name)

# %% id="-eD0b3u_mkDo"
chroma_db2 = Chroma.from_documents(documents=docs,
                                  collection_name='wikipedia_db2',
                                  embedding=mxbai_embedder,
                                  collection_metadata={"hnsw:space": "cosine"})

retriever2 = chroma_db2.as_retriever(search_type="similarity",
                                     search_kwargs={"k": 2})

# %% colab={"base_uri": "https://localhost:8080/"} outputId="bd5a1449-21b6-471c-be93-591b10c08766" id="zKFg28uunC7M"
query = "what is the capital of India?"
top_docs = retriever2.invoke(query)
top_docs

# %% colab={"base_uri": "https://localhost:8080/"} outputId="e909ee91-0738-4b88-83cc-39b7f3a181ea" id="aq-BGb7tnC7T"
query = "what is the old capital of India?"
top_docs = retriever2.invoke(query)
top_docs

# %% id="jzrIVI2NAHC1"
from langchain_openai import OpenAIEmbeddings

# details here: https://openai.com/blog/new-embedding-models-and-api-updates
openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')

# %% id="ST7PDWG2b0ln"
chroma_db3 = Chroma.from_documents(documents=docs,
                                   collection_name='wikipedia_db3',
                                   embedding=openai_embed_model,
                                   collection_metadata={"hnsw:space": "cosine"})

retriever3 = chroma_db3.as_retriever(search_type="similarity",
                                     search_kwargs={"k": 2})

# %% colab={"base_uri": "https://localhost:8080/"} outputId="7dc474ae-5fb5-40f4-f36c-75eca9a7a11e" id="A06_6YQ0ofXp"
query = "what is the capital of India?"
top_docs = retriever3.invoke(query)
top_docs

# %% colab={"base_uri": "https://localhost:8080/"} outputId="3c56288c-c78b-427f-e340-8a53e4834da0" id="C-6CvayKofXq"
query = "what is the old capital of India?"
top_docs = retriever3.invoke(query)
top_docs

# %% [markdown] id="IAiLM7_g5izD"
# # Exploring Advanced Retrieval & Reranking Strategies

# %% [markdown] id="6HKdIe2_wIBR"
# ### Multi Query Retrieval
#
# Retrieval may produce different results with subtle changes in query wording, or if the embeddings do not capture the semantics of the data well. Prompt engineering / tuning is sometimes done to manually address these problems, but can be tedious.
#
# The [`MultiQueryRetriever`](https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.multi_query.MultiQueryRetriever.html) automates the process of prompt tuning by using an LLM to generate multiple queries from different perspectives for a given user input query. For each query, it retrieves a set of relevant documents and takes the unique union across all queries to get a larger set of potentially relevant documents.

# %% id="qmfVRrRujxrA"
from langchain_openai import ChatOpenAI

chatgpt = ChatOpenAI(model_name="gpt-4o", temperature=0)

# %% id="s3gU4HPvfjs7"
from langchain.retrievers.multi_query import MultiQueryRetriever
# Set logging for the queries
import logging

similarity_retriever3 = chroma_db3.as_retriever(search_type="similarity",
                                                search_kwargs={"k": 2})

mq_retriever = MultiQueryRetriever.from_llm(
    retriever=similarity_retriever3, llm=chatgpt,
    include_original=True
)

logging.basicConfig()
# so we can see what queries are generated by the LLM
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# %% id="KZJBIsl8xC_N" colab={"base_uri": "https://localhost:8080/"} outputId="eb035db7-62a6-47a2-fe88-5cf6ec64b44c"
query = "what is the capital of India?"
docs = mq_retriever.invoke(query)
docs

# %% id="TVBeusuSyVo3" colab={"base_uri": "https://localhost:8080/"} outputId="dc05fec9-8b6b-4bf5-f5e5-1022ba3630e9"
query = "what is the old capital of India?"
docs = mq_retriever.invoke(query)
docs

# %% [markdown] id="9rWORsRu5npv"
# ### Hybrid Search (BM25 + Semantic)

# %% id="EZdxSkaX0W96"
laptops = [
    "The laptop model XPS13-9380 by Dell comes with an Intel Core i7-8565U processor, 16GB RAM, and 512GB SSD. It also features a 13.3-inch FHD display.",
    "Apple's MacBook Pro 16-inch model MVVJ2LL/A includes a 9th-generation Intel Core i9 processor, 16GB of RAM, and a 1TB SSD. It has a stunning Retina display with True Tone technology.",
    "The HP Spectre x360 15T-eb000 has an Intel Core i7-10750H processor, 16GB RAM, and a 512GB SSD. This model, 7DF22AV_1, also features a 4K UHD touch display.",
    "Lenovo's ThinkPad X1 Carbon Gen 8, part number 20U9005MUS, is equipped with an Intel Core i7-10510U, 16GB RAM, and a 1TB SSD. It is known for its lightweight design and durability.",
    "The ASUS ZenBook 14 UX434FL-DB77 features an Intel Core i7-8565U, 16GB RAM, and a 512GB SSD. This model also comes with a 14-inch FHD display and ScreenPad 2.0.",
    "Microsoft Surface Laptop 3 model VEF-00064 has an Intel Core i7-1065G7 processor, 16GB RAM, and a 256GB SSD. It is known for its sleek design and vibrant PixelSense display."
]

# %% id="euJ2SskG1Bc9"
laptop_db = Chroma.from_texts(laptops, collection_name='laptop_db',
                              embedding=openai_embed_model,
                              collection_metadata={"hnsw:space": "cosine"})

# %% id="rnp6J8F0brql"
from langchain.retrievers import BM25Retriever, EnsembleRetriever

similarity_retriever = laptop_db.as_retriever(search_type="similarity",
                                                search_kwargs={"k": 2})

bm25_retriever = BM25Retriever.from_texts(laptops)
bm25_retriever.k = 2

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, similarity_retriever],
    weights=[0.7, 0.3]
)


# %% colab={"base_uri": "https://localhost:8080/"} id="Z_vSZhhn1nxF" outputId="4e122753-99ae-46f2-d218-0c3de8ac3492"
query = "laptops with 16GB RAM and processor i7-1065G7 intel"
docs = similarity_retriever.invoke(query)
docs

# %% colab={"base_uri": "https://localhost:8080/"} id="rDEcDAdvzhEa" outputId="499810c2-67d5-4962-fa04-558aae015153"
query = "laptops with 16GB RAM and processor i7-1065G7 intel"
docs = ensemble_retriever.invoke(query)
docs

# %% [markdown] id="9YK8BBMBAU9o"
# ### Chained Retrieval with Reranker
#
# This strategy uses a chain of multiple retrievers sequentially to get to the most relevant documents. The following is the flow
#
# Similarity Retrieval → Reranker Model Retrieval

# %% id="UgZ4eMFc2GgI" colab={"base_uri": "https://localhost:8080/", "height": 177, "referenced_widgets": ["6fd7c44f3bbb4204bb2e07308e892271", "5546eb8afa764e03976a36bc86dac715", "46ff79c6f19644a985700cf0fd36e7e6", "b83e708ad86c4c839007d134ff936509", "57deee2cf26e4a80b0cd9feadf279067", "eb442a88ff1d4771b31b4d252fd5dcfc", "cbcd1eca97bc4b63b17bd4e6d9e8dd2a", "31480ece699743cd97865ab33daaa205", "223f12491de34df088c82cc05a1ca283", "731caf79269c49d7bfa384eb5997abf9", "066525a39ba2457c9bea8f26fa6c4df5", "d86b80c217a24ebf8bebdbdf5424ad1b", "ee711859d98f4d37a97e7035ffc602f4", "20455ca8ba2a4749b849d2d8c65835bc", "96ad10bc1d5b4ae4adee08f2bacad8ce", "1d3abb0b0904467db10f030e508c4812", "c56d3968538f439691eb521cf2e06d89", "e41d2be18fa1442abbecf728168a3f45", "517de951319b46c3b23b99f2ba0d6d55", "35fe9e01d6474a81903761ec16b58b34", "a0c9319da48040e790042490b3c7b5c0", "dbb11937fc9b47e08dd51af8334bf11e", "2c6e4e75f6834649884cccceff71127b", "e52776190d40478abfdde8d5211fb6cf", "fd007d8412f646f0861e6689f13c58c9", "69bb3e89f95244cf985fa23566ddc35e", "8ef7f8d06a734c0ab066ac97458bf65c", "55da1657e8114608b245ea701fa32ef4", "7b941b76cf9a42649c348bae42465fa7", "9f905168024940e99893907373986442", "807eb8cbdace4822ad8a19a5abd7a576", "6b41967bc1324a4dabe26a6519716d17", "009e98328a394a0c92349eb86462783c", "dbfdf37984c34af8930973d88bd0b95d", "0728b0e8039a462b9d437a8cabba2657", "5b3c6a4a6729441dacd6db24d2e961c5", "c4aaff4a294a4f62a5e75127b1122f62", "b0b60f019c79489491610cbb03d4da15", "db700efba39b4607beacb8b4bf8cd124", "388d91bd892a45708a60d9aedab534ef", "e58d8a664e6e406bbc81466fbd72094c", "f64f3a18dab34e09ba958d3783215194", "d735b8edb25d4460bf0460ddba5d339a", "da8f6a3b11964983a4711897d6b67d22", "20f6c8a5ec264253982055fc05614581", "06920f01f6de40629304615b8364962a", "8d0a45640659459c8fc89e06abb49d4b", "6f22ff6692af48a295c6527faa808306", "999e1838428b410690973dafd6737718", "56c6b2020f1f4e04b59a55b7278d80a4", "ff42800e905d49e78073f1504fcbb85a", "fe4a10c2d53a4e4c9abafb6374f2d584", "965570e09b3a4519b23eb198939075d0", "bfeabc1d92d7487b9f54ff5ff22d7a7e", "75bf667edcc8429091075f0340b46908"]} outputId="1bfcf5f6-3fd3-436d-90ae-3af202d2c736"
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever

# Retriever 1 - simple cosine distance based retriever
similarity_retriever = chroma_db3.as_retriever(search_type="similarity",
                                              search_kwargs={"k": 3})

# download an open-source reranker model - cross-encoder/qnli-electra-base
reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/qnli-electra-base")
reranker_compressor = CrossEncoderReranker(model=reranker, top_n=2)
# Retriever 2 - Uses a Reranker model to rerank retrieval results from the previous retriever
final_retriever = ContextualCompressionRetriever(
    base_compressor=reranker_compressor,
    base_retriever=similarity_retriever
)

# %% colab={"base_uri": "https://localhost:8080/"} outputId="4caf634d-fd3a-4cb2-8e5b-870338d260d8" id="0pv2xzUCvI-k"
query = "what is the capital of India?"
docs = final_retriever.invoke(query)
docs

# %% colab={"base_uri": "https://localhost:8080/"} outputId="763cf7b0-f386-4312-9e3a-4f12fc68226b" id="vwgN4uQSvI-l"
query = "what is the old capital of India?"
docs = final_retriever.invoke(query)
docs

# %% [markdown] id="T3tqkxqx50Kf"
# # Exploring Context Compression Strategies

# %% [markdown] id="nsljt2T6RfxU"
# ### LLM Prompt-based Contextual Compression Retrieval
#
# The context compression can happen in the form of:
#
# - Remove parts of the content of retrieved documents which are not relevant to the query. This is done by extracting only relevant parts of the document to the given query
#
# - Filter out documents which are not relevant to the given query but do not remove content from the document

# %% [markdown] id="BuqLaFoeRfxc"
# Here we look at `LLMChainExtractor`, which will iterate over the initially returned documents and extract from each only the content that is relevant to the query. Totally irrelevant documents might also be dropped

# %% id="08OzX1F_Rfxd"
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

chatgpt = ChatOpenAI(model_name="gpt-4o", temperature=0)

similarity_retriever = chroma_db3.as_retriever(search_type="similarity",
                                              search_kwargs={"k": 3})

# extracts from each document only the content that is relevant to the query
compressor = LLMChainExtractor.from_llm(llm=chatgpt)

# retrieves the documents similar to query and then applies the compressor
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=similarity_retriever
)

# %% colab={"base_uri": "https://localhost:8080/"} outputId="399d52a4-e4e8-42b6-a779-4c2c6e99128f" id="FtTkK13vRfxd"
query = "what is the capital of India?"
docs = compression_retriever.invoke(query)
docs

# %% colab={"base_uri": "https://localhost:8080/"} outputId="3cbbf415-58c7-4c47-aedc-0f9ee5a06708" id="pyYeM2N_Rfxd"
query = "what is the old capital of India?"
docs = compression_retriever.invoke(query)
docs

# %% [markdown] id="1izKdZtyRfxd"
# The `LLMChainFilter` is slightly simpler but more robust compressor that uses an LLM chain to decide which of the initially retrieved documents to filter out and which ones to return, without manipulating the document contents.

# %% id="12GSESJ4Rfxd"
from langchain.retrievers.document_compressors import LLMChainFilter

similarity_retriever = chroma_db3.as_retriever(search_type="similarity",
                                              search_kwargs={"k": 3})

#  decides which of the initially retrieved documents to filter out and which ones to return
_filter = LLMChainFilter.from_llm(llm=chatgpt)

# retrieves the documents similar to query and then applies the filter
compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter, base_retriever=similarity_retriever
)

# %% colab={"base_uri": "https://localhost:8080/"} outputId="449f13a6-d98b-49d8-a706-d4943e90496e" id="0h8XIL3uRfxd"
query = "what is the capital of India?"
docs = compression_retriever.invoke(query)
docs

# %% colab={"base_uri": "https://localhost:8080/"} outputId="5bcb8d78-fdfa-4d66-f843-d5f25cb15636" id="39CIHT3NRfxd"
query = "what is the old capital of India?"
docs = compression_retriever.invoke(query)
docs
