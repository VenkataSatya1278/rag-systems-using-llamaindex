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
# # Metadata Management

# %% [markdown] id="B0CMpBBJMPXN"
# In this notebook we will look into customizing documents with metadata.

# %% executionInfo={"elapsed": 500, "status": "ok", "timestamp": 1703423696702, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="7u6F6LEWLWtP"
# The nest_asyncio module enables the nesting of asynchronous functions within an already running async loop.
# This is necessary because Jupyter notebooks inherently operate in an asynchronous loop.
# By applying nest_asyncio, we can run additional async functions within this existing loop without conflicts.
import nest_asyncio
nest_asyncio.apply()

# %%
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv('D:/Training/FAA-Training/Beyond-the-Prompt-Practical-RAG-for-Real-World-AI/RAG-systems-using-LlamaIndex/RAG-System-Using-LamaIndex/.env')
# load_dotenv('D:/.env')
# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# %% [markdown] id="TJ_66T7F9WkN"
# # Download Data
#
# We will use Paul Graham essay text for this tutorial.

# %% executionInfo={"elapsed": 500, "status": "ok", "timestamp": 1703423696702, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="7u6F6LEWLWtP"
# !mkdir -p 'data/paul_graham/'
# https://github.com/run-llama/llama_index/tree/main/docs/docs/examples/data/paul_graham
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# %%

# %%

# %% [markdown] id="f90MHNjwMgGv"
# # Defining Documents
#
# Documents can either be created automatically via data loaders, or constructed manually.
#
# By default, all of our `data loaders` (including those offered on LlamaHub) return Document objects through the `load_data` function.

# %% executionInfo={"elapsed": 500, "status": "ok", "timestamp": 1703423857892, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="9IBNAY0FMCq5"
from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1703423710445, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="bbM_1ercMqXP" outputId="9f5bb391-a281-49a7-a246-591c08629521"
documents

# %%
len(documents)

# %%
documents[0].metadata

# %% [markdown] id="NR8vbI-hMzgW"
# If you look-up closely, you will see the text is loaded as single `Document` object. And there are certain metadata associated with it as well.

# %%

# %%

# %% [markdown] id="B-qsVH8LNHjv"
# # Customizing Documents
#
# This section covers various ways to customize `Document` objects. Since the `Document` object is a subclass of our `TextNode` object, all these settings and details apply to the `TextNode` object class as well.

# %% [markdown] id="GrAuQ-3cNZsd"
# ## Metadata
#
# Documents also offer the chance to include useful `metadata`. Using the metadata dictionary on each document, additional information can be included to help inform responses and track down sources for query responses. This information can be anything, such as filenames or categories. If you are integrating with a vector database, keep in mind that some vector databases require that the keys must be strings, and the values must be flat (either `str`, `float`, or `int`).
#
# Any information set in the `metadata` dictionary of each document will show up in the metadata of each source node created from the document. Additionally, this information is included in the nodes, enabling the index to utilize it on queries and responses. `By default, the metadata is injected into the text for both embedding and LLM model calls.`

# %% [markdown] id="EKhDSZn7NrA3"
# There are a few ways to set up this dictionary:

# %% [markdown] id="9hCCNmlbNtMw"
# ### In the document constructor:

# %% executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1703423770024, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="MvPKBRIxMvbN"
from llama_index.core import Document

# %%
document = Document(
    text=documents[0].text,
    metadata={"filename": "paul_graham_essay.txt", "category": "essay_text"},
)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1703423772764, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="nopUNua9N1hv" outputId="01ff571c-1bf0-4293-9d04-44b7abb771f1"
document

# %%
document.metadata

# %%

# %%

# %% [markdown] id="kSX6BtxfOB26"
# ### After the document is created:
#
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 505, "status": "ok", "timestamp": 1703423800831, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="cgxBIvXrOHIr" outputId="65adb0b8-c0f8-484d-a07f-00ad54019e85"
documents[0].metadata

# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1703423891359, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="YXEAwlw7N7U2"
# you can rename the filename in the metadata
documents[0].metadata["file_name"] = "paul_graham_essay_2021.txt"

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1703423893223, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="DZXi_pQEON9W" outputId="790bd6c5-f8fb-483b-cdd2-fdcb9c610455"
documents[0].metadata

# %% executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1703423900205, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="HM_MO8cwOPUT"
# Add the "category" key in metadata
documents[0].metadata["category"] = "biography"

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1703423903063, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="97f5frRLOiIg" outputId="a1630847-0222-4fec-c273-a4f668dc66f1"
documents[0].metadata

# %%

# %%

# %% [markdown]
# ### Using the SimpleDirectoryReader and file_metadata hook

# %% [markdown] id="PC4utZt2Om2n"
# Set the filename automatically using the SimpleDirectoryReader and file_metadata hook. This will automatically run the hook on each document to set the metadata field:

# %% executionInfo={"elapsed": 694, "status": "ok", "timestamp": 1703423935659, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="hIRqlIwqOjL_"
from llama_index.core import SimpleDirectoryReader

filename_fn = lambda filename: {"file_name": filename, "category": 'biography'}

# automatically sets the metadata of each document according to filename_fn
documents = SimpleDirectoryReader("./data/paul_graham", file_metadata=filename_fn).load_data()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1703423936169, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="20K61FIJPCu9" outputId="c2f02c56-7d4c-4173-f902-36dd269b4db3"
documents[0].metadata

# %%

# %%

# %% [markdown] id="MpnyWRuhP_W_"
# ## Advanced - Metadata Customization
#
# A key detail mentioned above is that by default, any metadata you set is included in the embeddings generation and LLM.

# %% [markdown] id="xjtUudnoQFCh"
# ### Customizing LLM Metadata Text
#
# Typically, a document might have many metadata keys, but you might not want all of them visible to the LLM during response synthesis. In the above examples, we may not want the LLM to read the file_name of our document. However, the file_name might include information that will help generate better embeddings. A key advantage of doing this is to bias the embeddings for retrieval without changing what the LLM ends up reading.
#
# We can exclude it like so:

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1703424015699, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="Fz6WIjrnQRS8" outputId="28643744-c8cf-47c0-ac72-8f0e3e48d2b9"
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import MetadataMode

documents = SimpleDirectoryReader("./data/paul_graham").load_data()

print(documents[0].get_content(metadata_mode=MetadataMode.LLM))

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1703424031991, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="x86SkbciQNlG" outputId="0078a5c8-b6b6-46e5-c0da-087d190cb51e"
documents[0].metadata.keys()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 514, "status": "ok", "timestamp": 1703424039871, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="XZrS5HkwQjNR" outputId="bfd6cc4c-cad3-47ad-bf68-8b315aa232a9"
documents[0].excluded_llm_metadata_keys

# %%

# %%

# %% [markdown] id="5IxBFEoSQ-sx"
# ### Add `category` as metadata

# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1703424076970, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="ppc3UovJRDg5"
documents[0].metadata['category'] = 'paul graham biography'

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1703424078681, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="kMkyjxdNRKc9" outputId="43389857-dab0-42c6-bae8-47ce9d60cb94"
documents[0].metadata

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 983, "status": "ok", "timestamp": 1703424083656, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="lPBd-gfjRPeG" outputId="9b4ad429-2cc5-41a4-a750-187638ee5e3d"
print(documents[0].get_content(metadata_mode=MetadataMode.LLM))

# %% executionInfo={"elapsed": 510, "status": "ok", "timestamp": 1703424102744, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="Tq3MmnppQ0GR"
documents[0].excluded_llm_metadata_keys = ['file_path', 'file_name', 'file_type', 'file_size', 'creation_date', 'last_modified_date', 'last_accessed_date', 'category']

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1703424105259, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="RlrsAckxRXt7" outputId="81d4a99f-f9fd-475b-b94e-bdc3a8b561e6"
print(documents[0].get_content(metadata_mode=MetadataMode.LLM))

# %%

# %%

# %% [markdown] id="JPZtBRzmSUcm"
# ### Customizing Embedding Metadata Text
#
# Similar to customing the metadata visible to the LLM, we can also customize the metadata visible to embeddings. In this case, you can specifically exclude metadata visible to the embedding model, in case you DON’T want particular text to bias the embeddings.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 510, "status": "ok", "timestamp": 1703424144021, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="hoB0NSh2RdNb" outputId="368f648d-2aba-4079-e744-ce6d6e72d4a5"
documents[0].excluded_embed_metadata_keys

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1703424147807, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="-YQwg2cRTNfL" outputId="52a5085e-c1d8-4dbd-b12d-44b0300027af"
documents[0].metadata.keys()

# %% [markdown] id="ev9VdOimSd7q"
# Then, we can test what the embedding model will actually end up reading using the get_content() function and specifying MetadataMode.EMBED:

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 762, "status": "ok", "timestamp": 1703424174111, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="8jYW9XE1SdEU" outputId="49eb08a3-3bde-41a4-8edc-86caec2a48af"
print(documents[0].get_content(metadata_mode=MetadataMode.EMBED))

# %% executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1703424193603, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="xx_MW4raSZtG"
documents[0].excluded_embed_metadata_keys = ['file_path',
                                             'file_type',
                                             'file_size',
                                             'creation_date',
                                             'last_modified_date',
                                             'last_accessed_date',
                                             'category']

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1703424196193, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="IUqK2AIpTETI" outputId="f8fcae03-7270-4964-8a77-5becdcc3dadc"
print(documents[0].get_content(metadata_mode=MetadataMode.EMBED))

# %%

# %%

# %% [markdown] id="CsMYB_OTTXRQ"
# ### Customizing Metadata Format
#
# As you know by now, metadata is injected into the actual text of each document/node when sent to the LLM or embedding model. By default, the format of this metadata is controlled by three attributes:
#
# 1. Document.metadata_seperator -> default = "\n"
#
# When concatenating all key/value fields of your metadata, this field controls the separator between each key/value pair.
#
# 2. Document.metadata_template -> default = "{key}: {value}"
#
# This attribute controls how each key/value pair in your metadata is formatted. The two variables key and value string keys are required.
#
# 3. Document.text_template -> default = {metadata_str}\n\n{content}
#
# Once your metadata is converted into a string using metadata_seperator and metadata_template, this templates controls what that metadata looks like when joined with the text content of your document/node. The metadata and content string keys are required.

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 556, "status": "ok", "timestamp": 1703424266388, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="zvA9-q0CTWyY" outputId="1223f04e-d722-4e5d-caf1-88922868cc6a"
documents[0].text_template

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 514, "status": "ok", "timestamp": 1703424271809, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="ZOS0T9YKTG2Z" outputId="45743f1c-402b-4177-d1f7-f9a9746f4705"
## You can change it

documents[0].text_template = '{metadata_str}\n{content}'

documents[0].text_template

# %%

# %%

# %% [markdown] id="HQWhSV8kUD-3"
# # End to End example

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1703424327187, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="z4tNl5N9UEqp" outputId="2f8cedd7-e9ca-485f-dbe3-e69a13556868"
from llama_index.core import Document
from llama_index.core.schema import MetadataMode

document = Document(
    text="This is a super-customized document",
    metadata={
        "file_name": "super_secret_document.txt",
        "category": "finance",
        "author": "LlamaIndex",
    },
    excluded_llm_metadata_keys=[], # we will include all metadata to be seen by LLM.
    excluded_embed_metadata_keys=['author'], # we will include all metadata except author name to be used for embedding
    # metadata_seperator="::",
    # metadata_template="{key} -> {value}",
    text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1703424327187, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="z4tNl5N9UEqp" outputId="2f8cedd7-e9ca-485f-dbe3-e69a13556868"
print(
    "The LLM sees this: \n",
    document.get_content(metadata_mode=MetadataMode.LLM),
)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1703424327187, "user": {"displayName": "Ravi Theja", "userId": "12148656718425770960"}, "user_tz": -330} id="z4tNl5N9UEqp" outputId="2f8cedd7-e9ca-485f-dbe3-e69a13556868"
print(
    "The Embedding model sees this: \n",
    document.get_content(metadata_mode=MetadataMode.EMBED),
)

# %%
