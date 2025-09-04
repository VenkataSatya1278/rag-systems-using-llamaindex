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
# <a href="https://colab.research.google.com/github/dipanjanS/improving-RAG-systems-dhs2024/blob/main/Demo_1_Solutions_for_Missing_Content_in_RAG.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="b8kkhS-UgJAR"
# # Solutions for Missing Content in RAG
#
# Here we will explore the following strategies
#
# - Better Data Cleaning
# - Better Prompting
# - Agentic RAG with Tools
#

# %% [markdown] id="mjyiD8I5r7jB"
# ## Exploring Useful Data Loaders

# %% [markdown] id="L1KvMtf54l0d"
# #### Install OpenAI, HuggingFace and LangChain dependencies

# %% id="2evPp14fy258" colab={"base_uri": "https://localhost:8080/"} outputId="794909d6-9d36-4e1a-ecc3-e23c5ee08257"
# !pip install langchain
# !pip install langchain-openai
# !pip install langchain-community

# %% id="CB6lHzbz5a10" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="6bdb528c-ae6f-4177-c893-77b923a69041"
# takes 2 - 5 mins to install on Colab
# !pip install "unstructured[all-docs]"

# %% id="NhEW-tOywUgt" colab={"base_uri": "https://localhost:8080/"} outputId="80dd00ff-8481-4d70-b796-01d669382ebd"
# install OCR dependencies for unstructured
# !sudo apt-get install tesseract-ocr
# !sudo apt-get install poppler-utils

# %% id="MWNjOhSbRaOw" colab={"base_uri": "https://localhost:8080/"} outputId="adc9f207-3873-4ff7-d556-b435babb1fbf"
# !pip install pymupdf==1.24.4

# %% [markdown] id="tZB2fxI9fKxC"
# #### PDF Loaders
#
# [Portable Document Format (PDF)](https://en.wikipedia.org/wiki/PDF), standardized as ISO 32000, is a file format developed by Adobe in 1992 to present documents, including text formatting and images, in a manner independent of application software, hardware, and operating systems.
#
# LangChain integrates with a host of PDF parsers. Some are simple and relatively low-level; others will support OCR and image-processing, or perform advanced document layout analysis. The right choice will depend on your use-case and through experimentation.
#
# Here we will see how to load PDF documents into the LangChain `Document` format
#
# We download a research paper to experiment with

# %% [markdown] id="p2NWiC51KDbm"
# If the following command fails you can download the paper manually by going to http://arxiv.org/pdf/2103.15348.pdf, save it as `layoutparser_paper.pdf`and upload it on the left in Colab from the upload files option

# %% id="t_zMe1cES7Tb" colab={"base_uri": "https://localhost:8080/"} outputId="2b0168af-d454-4bcb-e398-6b323a2291df"
# !wget -O 'layoutparser_paper.pdf' 'http://arxiv.org/pdf/2103.15348.pdf'

# %% [markdown] id="x_T0_7KtfUJj"
# #### PyMuPDFLoader
#
# This is the fastest of the PDF parsing options, and contains detailed metadata about the PDF and its pages, as well as returns one document per page. It uses the `pymupdf` library internally.

# %% id="l3KTMV_3XmfL"
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("./layoutparser_paper.pdf")
pages = loader.load()

# %% id="JXUAxQnJhuHO" colab={"base_uri": "https://localhost:8080/"} outputId="bc6d42cb-f4e0-4001-a2c2-195f2ffc5c3c"
len(pages)

# %% id="CZM5poERdRpL" colab={"base_uri": "https://localhost:8080/"} outputId="92186e6a-555e-426e-c0d1-31ed0b66570e"
pages[0]

# %% id="Uhr0Y1C90TH0" colab={"base_uri": "https://localhost:8080/"} outputId="10e4f7ad-ec77-4e6d-ff6b-eb4723b9e75a"
pages[0].metadata

# %% id="BdaA9hkHXmhs" colab={"base_uri": "https://localhost:8080/"} outputId="c11f09eb-9162-43dd-84a4-409be133302a"
print(pages[0].page_content)

# %% id="eDxTzEe0Le6I" colab={"base_uri": "https://localhost:8080/"} outputId="807e4ce2-32c6-43dd-8de9-aaadf706df0d"
print(pages[4].page_content)

# %% [markdown] id="3Zvyk9ACL8fx"
# #### UnstructuredPDFLoader
#
# [Unstructured.io](https://unstructured-io.github.io/unstructured/) supports a common interface for working with unstructured or semi-structured file formats, such as Markdown or PDF. LangChain's [`UnstructuredPDFLoader`](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.pdf.UnstructuredPDFLoader.html) integrates with Unstructured to parse PDF documents into LangChain [`Document`](https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html) objects.

# %% id="Z8qz1HKWMRTf"
from langchain_community.document_loaders import UnstructuredPDFLoader

# %% [markdown] id="CeB_kHTjR6u5"
# Load PDF with complex parsing, table detection and chunking by sections

# %% id="pdKdca4RMmY8"
# takes 3-4 mins on Colab
loader = UnstructuredPDFLoader('./layoutparser_paper.pdf',
                               strategy='hi_res',
                               extract_images_in_pdf=True,
                               infer_table_structure=True,
                               chunking_strategy="by_title", # section-based chunking
                               max_characters=4000, # max size of chunks
                               new_after_n_chars=3800, # preferred size of chunks
                               combine_text_under_n_chars=2000, # smaller chunks < 2000 chars will be combined into a larger chunk
                               mode='elements')
data = loader.load()

# %% id="0WvtzXCgMots" colab={"base_uri": "https://localhost:8080/"} outputId="ce118316-412b-4d2a-9ad1-efa456902204"
len(data)

# %% id="JM5fCpJNPPiv" colab={"base_uri": "https://localhost:8080/"} outputId="e887db04-f080-46b2-dc9d-c66a80b4a34b"
[doc.metadata['category'] for doc in data]

# %% id="9f7zcYUxMq0t" colab={"base_uri": "https://localhost:8080/"} outputId="e04d08b5-2c5b-4f0b-fff9-a5ba04f5344a"
data[0]

# %% id="XdjS77Fh0oKJ" colab={"base_uri": "https://localhost:8080/"} outputId="64895ec2-2d90-4c12-9875-b8160d4f5269"
print(data[0].page_content)

# %% id="six4KwFGPX-S" colab={"base_uri": "https://localhost:8080/"} outputId="264f3a1d-d692-466d-cbc8-ecb7a2cb48e3"
data[5]

# %% id="AYe9e5K3Pfje" colab={"base_uri": "https://localhost:8080/", "height": 88} outputId="e0833c02-7aca-434e-c524-4e0aae985dab"
data[5].page_content

# %% id="nFI5BBDjPoHK" colab={"base_uri": "https://localhost:8080/", "height": 147} outputId="10fead30-7a01-48e6-84fa-8eb9c3b41c4a"
from IPython.display import HTML

HTML(data[5].metadata['text_as_html'])

# %% colab={"base_uri": "https://localhost:8080/"} id="mgU18vNXCNEY" outputId="d03d389f-673f-494d-ff00-a37ae2d39971"
# ! ls -l ./figures

# %% colab={"base_uri": "https://localhost:8080/", "height": 345} id="vh_zj91eB3Ff" outputId="de6d3bde-8929-4710-fda0-d050582659c6"
from IPython.display import Image

Image('./figures/figure-4-1.jpg')

# %% [markdown] id="QQV0X4grW971"
# #### Microsoft Office Document Loaders
#
# The Microsoft Office suite of productivity software includes Microsoft Word, Microsoft Excel, Microsoft PowerPoint, Microsoft Outlook, and Microsoft OneNote. It is available for Microsoft Windows and macOS operating systems. It is also available on Android and iOS.
#
# [Unstructured.io](https://docs.unstructured.io/open-source/introduction/overview) provides a variety of document loaders to load MS Office documents. Check them out [here](https://docs.unstructured.io/open-source/core-functionality/partitioning).
#
# Here we will leverage LangChain's [`UnstructuredWordDocumentLoader`](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.word_document.UnstructuredWordDocumentLoader.html) to load data from a MS Word document.

# %% id="V37elBlCT6Mn" colab={"base_uri": "https://localhost:8080/"} outputId="26f1309c-c0b6-4b97-e7ee-c1ea4db3b496"
# !gdown 1DEz13a7k4yX9yFrWaz3QJqHdfecFYRV-

# %% id="BqYBSoqHT6TY"
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

# %% [markdown] id="ILz-pXIyX8e0"
# Load word doc with complex parsing and section based chunks

# %% id="7cKYWNjAUo1A"
loader = UnstructuredWordDocumentLoader('./Intel Strategy.docx',
                                        strategy='fast',
                                        chunking_strategy="by_title",
                                        max_characters=4000, # max limit of a document chunk
                                        new_after_n_chars=3800, # preferred document chunk size
                                        mode='elements')
data = loader.load()

# %% id="CkOn6t0QVOyE" colab={"base_uri": "https://localhost:8080/"} outputId="96612d50-0a7c-463e-ea1d-0ed157e63926"
len(data)

# %% id="IV4cp6yqVQZ1" colab={"base_uri": "https://localhost:8080/"} outputId="936a181a-a394-4ca4-9100-f3b80f2993d7"
data[0]

# %% id="mW5UhN9BVTaE" colab={"base_uri": "https://localhost:8080/"} outputId="219ce832-785b-4d17-b1ec-763cedafda45"
print(data[0].page_content)

# %% [markdown] id="U3hZ_O32as0x"
# ## Better Prompting for Consistent Results

# %% [markdown] id="H9c37cLnSrbg"
# #### Enter Open AI API Key

# %% id="cv3JzCEx_PAd" colab={"base_uri": "https://localhost:8080/"} outputId="5a47a66d-49c8-4893-f71e-409c69116ca5"
from getpass import getpass

OPENAI_KEY = getpass('Enter Open AI API Key: ')

# %% id="zT8VTBilFwQB"
import os

os.environ['OPENAI_API_KEY'] = OPENAI_KEY

# %% [markdown] id="2oeckxFBcc0E"
# #### Load Connection to LLM
#
# Here we create a connection to ChatGPT to use later in our chains

# %% id="vHa9LMOfcOCV"
from langchain_openai import ChatOpenAI

chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

# %% [markdown] id="htTiJ8RTgiAG"
# #### Problematic RAG Prompt

# %% id="JcHrMo7tF2dR"
from langchain_core.prompts import ChatPromptTemplate

prompt = """You are an assistant for question-answering tasks.
            Give an answer to the following question with the context provided

            Question:
            {question}

            Context:
            {context}

            Answer:
         """

prompt_template = ChatPromptTemplate.from_template(prompt)

# %% id="nfR1L5bKGA6Z"
context = """
The cricket world cup champion of 2023 is Australia

The cricket Twenty20 International champion of 2024 is India
"""

question = "Who is the euro 2020 champion?"

qa_rag_chain = (
    prompt_template
      |
    chatgpt
)

# %% colab={"base_uri": "https://localhost:8080/"} id="NCZARTPXGapG" outputId="6f0b80c3-b036-4a06-bffe-d0298d940e38"
response = qa_rag_chain.invoke({'context': context, 'question': question})
print(response.content)

# %% [markdown] id="uVc9olR5gmP6"
# #### Better RAG Prompt

# %% id="6VhrwTZcHKed"
from langchain_core.prompts import ChatPromptTemplate

prompt = """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If no context is present or if you don't know the answer, just say that you don't know.
            Do not make up the answer unless it is there in the provided context.

            Question:
            {question}

            Context:
            {context}

            Answer:
         """

prompt_template = ChatPromptTemplate.from_template(prompt)

# %% id="5_oTcrZAHbdv"
qa_rag_chain = (
    prompt_template
      |
    chatgpt
)

# %% colab={"base_uri": "https://localhost:8080/"} id="DBT4WW12HZ_j" outputId="8c37a9f3-145c-43e0-dafc-a32aa4f11a91"
response = qa_rag_chain.invoke({'context': context, 'question': question})
print(response.content)

# %% [markdown] id="DeLmkbOIaoWF"
# ## Agentic RAG with Tools

# %% [markdown] id="ucWRRI3QztL2"
# #### Enter Tavily Search Tool API Key
#
# Get a free API key from [here](https://tavily.com/#api)

# %% colab={"base_uri": "https://localhost:8080/"} id="mK-1WLzOrJdb" outputId="a33b7cd6-b390-4a75-8cd1-c0ecd2daa95d"
TAVILY_API_KEY = getpass('Enter Tavily Search API Key: ')

# %% id="R2h8vdQNVNjZ"
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

# %% [markdown] id="GQ36nHepgrgp"
# #### Setup Search Tool

# %% id="ALfCdvXqVO09"
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import rich

@tool
def search_web(query: str) -> list:
    """Search the web for a query."""
    tavily_tool = TavilySearchResults(max_results=3,
                                      search_depth='advanced',
                                      max_tokens=10000)
    results = tavily_tool.invoke(query)
    return [doc['content'] for doc in results]


# %% colab={"base_uri": "https://localhost:8080/", "height": 35} id="N7jzQYr7VesD" outputId="504af0ea-d30c-4659-f058-3db478b6a593"
question

# %% colab={"base_uri": "https://localhost:8080/"} id="bBuA2jXoVdL_" outputId="8027e5ea-cf3b-41a5-bbf1-1de633eec5f1"
search_web(question)

# %% [markdown] id="k4Khdw6Agvrn"
# #### Bind Tools to LLM

# %% id="4hCzYX4lVguE"
tools = [search_web]
chatgpt = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
chatgpt_with_tools = chatgpt.bind_tools(tools)

# %% [markdown] id="m2TlF4AwgyJT"
# #### Better RAG Prompt with Tool Calling

# %% id="X-5xXrygWyLL"
prompt = """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If no context is present or if you don't know the answer,
            check and see if you can use the tools available to you to get the answer.

            Question:
            {question}

            Context:
            {context}

            Answer:
         """

prompt_template = ChatPromptTemplate.from_template(prompt)

qa_rag_chain = (
    prompt_template
      |
    chatgpt_with_tools
)

# %% colab={"base_uri": "https://localhost:8080/"} id="pYUMXThpW8Bs" outputId="6ea92733-1472-46cc-ad1b-fe1bf6d5d88d"
context = """
The cricket world cup champion of 2023 is Australia

The cricket Twenty20 International champion of 2024 is India
"""

question = "Who is the euro 2020 champion?"

qa_rag_chain.invoke({'context': context, 'question': question})

# %% colab={"base_uri": "https://localhost:8080/"} id="HvmXXv_YXQgc" outputId="3a634cdb-39ed-46ab-c929-816f22583dd1"
question = "Who is the T20 2024 champion?"

qa_rag_chain.invoke({'context': context, 'question': question})


# %% [markdown] id="mGB14NL5g94n"
# #### Simple Agentic RAG with Tool Calls

# %% id="VzsRkX2AXGgl"
def agentic_rag(question, context):
  tool_call_map = {'search_web' : search_web}
  response = qa_rag_chain.invoke({'context': context, 'question': question})

  # if response content is present then we have our answer
  if response.content:
    print('Answer is in retrieved context')
    answer = response.content

  # if no response content present then call search tool
  elif response.tool_calls:
    print('Answer not in context, trying to use tools')
    tool_call = response.tool_calls[0]
    selected_tool = tool_call_map[tool_call["name"].lower()]
    print(f"Calling tool: {tool_call['name']}")
    tool_output = selected_tool.invoke(tool_call["args"])
    context = '\n\n'.join(tool_output)
    response = qa_rag_chain.invoke({'context': context, 'question': question})
    answer = response.content

  # no answer found from web search also
  else:
    answer = 'No answer found'

  print(answer)


# %% colab={"base_uri": "https://localhost:8080/"} id="KBm8mJSpXjNd" outputId="e423a026-e72c-4442-8c70-4a5de553454c"
context = """
The cricket world cup champion of 2023 is Australia

The cricket Twenty20 International champion of 2024 is India
"""

question = "Who is the T20 2024 champion?"

agentic_rag(question, context)

# %% colab={"base_uri": "https://localhost:8080/"} id="U-Zdn-quXjhY" outputId="e016abc7-7958-4b53-f17d-7c10f32da117"
question = "Who is the euro 2024 champion?"

agentic_rag(question, context)
