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

# %% [markdown]
# ## Content
# - How do you choose the right chunk size for your use case?
# - Chunkviz app https://chunkviz.up.railway.app/ 
# - Subword Tokenization  isa a new techinique in LLM whch is different technic from NLP.
# - Differnet types of subwork tokenization - Byte Pair encoding, WordPiece and SentencePiece.
# - Tokenization Estimation website - https://platform.openai.com/tokenizer 
# - Why is it important to know the total number of tokens in a chunk or document in RAG system?
# - Embedding models, their pricing and their token size
# - Text Generation models, their pricing and their token size

# %%
# !pip install dotenv
# !pip install llama-index-readers-file llama-index-readers-web unstructured

# %% [markdown]
# # Chunking and Token Counting Demo

# %% id="cKlax-updNW-"
import os

# %%
from dotenv import load_dotenv, find_dotenv

# %%
load_dotenv('D:/Training/FAA-Training/Beyond-the-Prompt-Practical-RAG-for-Real-World-AI/RAG-systems-using-LlamaIndex/RAG-System-Using-LamaIndex/.env')

# %% [markdown]
# ### File Loader

# %%
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter

# %% [markdown]
# https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt

# %%
documents = SimpleDirectoryReader(input_files=["./data/paul_graham_essay.txt"], filename_as_id=True).load_data(show_progress=True)

# %%
len(documents)

# %%
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

# %%
nodes = splitter.get_nodes_from_documents(documents)

# %%
len(nodes)

# %%
nodes[0]

# %%
nodes[0].metadata

# %%
print(nodes[0].text)

# %%
import tiktoken
for i, node in enumerate(nodes):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    print(f'number of tokens in node {i+1} - {len(encoding.encode(node.text))}')

# %%
print(nodes[0].text)

# %%
encoding.encode(nodes[0].text)

# %%
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

# %%
llm = OpenAI(model='gpt-4o-mini')

# %%
messages = [
    ChatMessage(
        role="system", content="answer the question accurately"
    ),
    ChatMessage(role="user", content="why is the sky blue?"),
]

# %%
response = llm.chat(messages)

# %%
response.message.content

# %%
response.raw

# %%
response.raw['usage'].dict()

# %%
len(encoding.encode(response.message.content))   # totak number of tokens in the output/response

# %%
len(response.message.content)  # number of characters in the response

# %% [markdown]
# ## Gettting the Token IDs

# %% [markdown]
# **By default, LlamaIndex uses a global tokenizer for all token counting. This defaults to `cl100k` from tiktoken, which is the tokenizer to match the default LLM `gpt-3.5-turbo.`**

# %%
enc = tiktoken.get_encoding("cl100k_base")

# %%
enc.encode("hello world!")


# %%
def count_tokens(text: str, encoding_name: str) -> int:
    """Counts the number of tokens in a text string."""
    
    encoding = tiktoken.get_encoding(encoding_name)  # Get the encoding based on the given name
    tokens = encoding.encode(text)  # Convert the text into tokens
    return len(tokens)  # Return the number of tokens


# %%
count_tokens("hello world!", 'cl100k_base')
