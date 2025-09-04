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
# # LLMs

# %% [markdown]
# ### Content
# - How to choose the right LLM? open source/closed source
#     - Select the domain specific LLM
#     - Select the state of the art LLM
#     - Mixtral of Experts
#     - 
# - 

# %% [markdown] id="wywCFaDQ39ic"
# In this notebook we will see how to interact with different LLMs through LlamaIndex.
#

# %% [markdown]
# Download the required packages by executing the below commands in either Anaconda Prompt (in Windows) or Terminal (in Linux or Mac OS)

# %% [markdown]
# pip install llama-index-llms-openai llama-index-llms-anthropic

# %%
# !pip install llama-index-llms-openai llama-index-llms-anthropic

# %% id="cKlax-updNW-"
import os

# %% id="cKlax-updNW-"
from dotenv import load_dotenv, find_dotenv
load_dotenv('D:/Training/FAA-Training/Beyond-the-Prompt-Practical-RAG-for-Real-World-AI/RAG-systems-using-LlamaIndex/RAG-System-Using-LamaIndex/.env')

# %%
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
HUGGINGFACE_API_KEY = os.environ['HUGGINGFACE_API_KEY']
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

# %% [markdown] id="8il00G0w6HBA"
# # OpenAI's LLM

# %% colab={"base_uri": "https://localhost:8080/", "height": 488} id="yxPcwFUQ6DKY" outputId="49910bb8-574c-4119-aebb-33f66d51888c"
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

# %%
llm = OpenAI(model='gpt-4o-mini') 

# %% [markdown] id="vN7jmOuH7SBs"
# ### Call complete with a prompt

# %% colab={"base_uri": "https://localhost:8080/", "height": 197} id="yREsde0o7NPT" outputId="d22ae3d3-3301-41fc-ceb3-305472b668f3"
# Use the model to complete a prompt
response = llm.complete("Write a welcome mail to the community members of Analytics Vidhya.")

# Print the generated response
print(response)

# %%
# raw response from the LLM
response.raw

# %%
# get the raw dict keys from the response
type(response.raw)

# %%
response.raw.__dict__.keys() 

# %%
# get the tokens usage from the raw response (raw is a ChatCompletion object)
response.raw.usage

# %%
response.raw.usage.model_dump()["completion_tokens"]

# %%
response.raw.usage.model_dump()["total_tokens"]

# %% [markdown] id="_R4dvP787gT6"
# ### Call chat with a list of messages.

# %% colab={"base_uri": "https://localhost:8080/", "height": 250} id="nwiWxJM97YaY" outputId="cf99685c-8edc-4da6-b497-0b50c0dbd313"
messages = [
    ChatMessage(
        role="system", content="You are an AI assistant that talks like Elon Musk."
    ),
    ChatMessage(role="user", content="Write a welcome mail to the community members of Analytics Vidhya."),
]

response = OpenAI().chat(messages)
print(response)

# %%
# convert the response into dict and fetch the keys
response.model_dump().keys()

# %%
# fetch the "content" from the message attribute
response.message.content

# %%
print(response.message.content)

# %%
len(response.message.content)   # total number of characters in the response message

# %%
# fetch the "raw" response from the LLM
response.raw

# %%
# get the tokens usage stats from the raw response
response.raw.usage.model_dump()

# %%
# get the completion_tokens, prompt_tokens & total_tokens form usage
response.raw.usage.model_dump()["prompt_tokens"]

# %%
response.raw.usage.model_dump()["total_tokens"]

# %% [markdown] id="sxNHcLYYW9m3"
# ### Using stream_chat endpoint.

# %% [markdown]
# The main difference between `chat` and `stream_chat` is how they handle the response from the AI model. The previous code waits for the full response before printing it, while the `stream_chat` streams the response in chunks and prints each chunk as it is received. This can provide a more interactive experience, especially for longer texts.

# %% colab={"base_uri": "https://localhost:8080/"} id="p0wqjnNwWwe_" outputId="dc4fa5c9-5ff2-4da3-ca6a-ead95d3e9b46"
messages = [
    ChatMessage(
        role="system", content="You are an AI assistant that talks like Master Yoda from the Star Wars."
    ),
    ChatMessage(role="user", content="Write a welcome mail to the community members of Analytics Vidhya."),
]

response = llm.stream_chat(messages)

for r in response:
  print(r.delta, end="")

# %% [markdown] id="0oA1GT1MXQv9"
# ### Configure Model

# %% id="_mtCmLwmXKox"
llm = OpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=100)

# %% id="YqaxpLd4xCAf"
response = llm.complete("Write a welcome mail to the community members of Analytics Vidhya.")
print(response)

# %%
llm = OpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=100,  additional_kwargs={"seed": 12345678, "top_p": 0.5})
response = llm.complete("Explain the concept of gravity in one sentence")
print(response)

# %% [markdown] id="b2lBstRkXaSP"
# #### Call chat with a list of messages.

# %% colab={"base_uri": "https://localhost:8080/"} id="inABtoqCXaSP" outputId="96afc602-1dd8-4a94-f0e5-6410306baa9e"
messages = [
    ChatMessage(
        role="system", content="You are an AI assistant that talks like Elon Musk."
    ),
    ChatMessage(role="user", content="Write a welcome mail to the community members of Analytics Vidhya."),
]

response = llm.chat(messages)
print(response)

# %%
response.model_dump().keys()

# %%
print(response.message.content)

# %%
len(response.message.content)

# %% [markdown] id="Z76JbeZgvLUO"
# ## Using Anthropic LLM API (Paid model)

# %% id="zl7lxVjzvOSY"
from llama_index.llms.anthropic import Anthropic

# %%
llm = Anthropic(model='claude-3-5-sonnet-20241022')

# %% id="ptiMqr4LvOUq"
response = llm.complete("Write a welcome mail to the community members of Analytics Vidhya.")
print(response)

# %% id="xkQUCVTHxEn9"
messages = [
    ChatMessage(
        role="system", content="You are an AI assistant that talks like a elon musk."
    ),
    ChatMessage(role="user", content="Write a welcome mail to the community members of Analytics Vidhya."),
]

response = llm.chat(messages)
print(response)
