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
# # Node Parsers

# %% [markdown]
# #### content
# - Types of node parsers -> TextNodeParser, HTMLNodeParser,JSONNodeParser, CSVNodeParser, MarkdownNodeParser, HierarchicalNodeParser, SimpleFileNodeParser,SimpleNodeParser.
# - Source Data -> Data Loaders -> Ingestion Pipeline (NodeParser + Custom Trasformation) -> Nodes (Node1, Node2, Node3 etc).
# - 

# %% id="cKlax-updNW-"
import os

# %%
from dotenv import load_dotenv, find_dotenv

# %%
# Load environment variables from the .env file
load_dotenv('D:/Training/FAA-Training/Beyond-the-Prompt-Practical-RAG-for-Real-World-AI/RAG-systems-using-LlamaIndex/RAG-System-Using-LamaIndex/.env')

# %%
# Retrieve the OpenAI API key from environment variables
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# %% [markdown]
# Download the required packages by executing the below commands in either Anaconda Prompt (in Windows) or Terminal (in Linux or Mac OS)

# %% [markdown]
# pip install tree-sitter_languages tree-sitter==0.21.3

# %% [markdown]
# https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/

# %%
# !pip install tree-sitter_languages tree-sitter==0.21.3

# %%
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# # Creating Nodes Manually

# %%
from llama_index.core import Document
from llama_index.core.schema import TextNode
doc = Document(text="This is a sample document text")

# %%
doc.doc_id

# %%
doc

# %%
doc.metadata

# %%
doc.metadata = {"report_name": "Competition Analysis Report May 2024",\
                     "department": "Marketing",\
                     "author": "Prashant"}

# %%
doc.to_dict

# %%
doc

# %%
# importing necessary classes
from llama_index.core import Document
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

# Creating a document
doc = Document(text="First sentence. Second Sentence")

# Creating text nodes
n1 = TextNode(text="First sentence", node_id=doc.doc_id)
n2 = TextNode(text="Second sentence", node_id=doc.doc_id)

# Setting up relationships
n1.relationships[NodeRelationship.NEXT] = n2.node_id
n2.relationships[NodeRelationship.PREVIOUS] = n1.node_id

# Displaying nodes and their relationships
print(n1.relationships)
print(n2.relationships)

# %%
#converting to dictionary
n1.dict()

# %%
#print text of n1
n1.text

# %% [markdown]
# In this example, we’ve manually created two Nodes and defined a previous or next relationship between them. The relationship tracks the order of Nodes within the original Document. This code tells LlamaIndex that the two Nodes belong to the initial Document and they also come in a particular order.

# %% [markdown]
# # File-Based Node Parsers

# %% [markdown]
# ## HTML Parser

# %%
import requests

# %%
from llama_index.core import Document

# %%
from llama_index.core.node_parser import HTMLNodeParser

# %% [markdown]
# - This parser uses Beautiful Soup to parse HTML files and convert them into nodes based on selected HTML tags. 
# - This parser simplifies the HTML file by extracting text from standard text elements and merging adjacent nodes of the same type.

# %%
# URL of the website to fetch HTML from
url = "https://docs.llamaindex.ai/en/stable/"

# Send a GET request to the URL
response = requests.get(url)
print(response)

# %%
# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Extract the HTML content from the response
    html_doc = response.text
    
    # Create a Document object with the HTML content
    document = Document(id_=url, text=html_doc)
    
    # Initialize the HTMLNodeParser with optional list of tags
    parser = HTMLNodeParser()
    
    # Parse nodes from the HTML document
    nodes = parser.get_nodes_from_documents([document])
    
else:
    # Print an error message if the request was unsuccessful
    print("Failed to fetch HTML content:", response.status_code)

# %%
len(nodes)

# %%
nodes[1].to_dict()

# %% [markdown]
# You have the option to customize the HTML tags from where you want to retrieve content.

# %%
my_tags = ["p", "span"]
html_parser = HTMLNodeParser(tags=my_tags)
nodes = html_parser.get_nodes_from_documents([document])

# %%
print('<span> elements:')
for node in nodes:
    if node.metadata['tag']=='span':
        print(node.text)

# %%

# %%
print('<p> elements:')
for node in nodes:
    if node.metadata['tag']=='p':
        print(node.text)

# %%

# %% [markdown]
# ## Simple File Node Parser

# %% [markdown]
# - This one automatically decides which of the following three node parsers should be used based on file types. 
# - It can automatically handle these file formats and transform them into nodes, simplifying the process of interacting with various types of content.

# %%
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path

# %% [markdown]
# Works for json, markdown, and html files

# %%
documents = FlatReader().load_data(Path('data/README.md'))

# %% [markdown]
# You can simply rely on `FlatReader` to load the file into your `Document` object; `SimpleFileNodeParser` will know what to do from there.

# %%
parser = SimpleFileNodeParser()

# %%
nodes = parser.get_nodes_from_documents(documents)

# %%
len(nodes)

# %%
nodes[5].to_dict()

# %%

# %%

# %% [markdown]
# # Text Splitters

# %% [markdown]
# ## Code splitter

# %% id="gGfPPk4gBAkQ"
from llama_index.core import SimpleDirectoryReader

# %%
from llama_index.core.node_parser import CodeSplitter

# %%
documents = SimpleDirectoryReader(input_files=['data/settings.py']).load_data()

# %%
splitter = CodeSplitter(
    language="python",
    chunk_lines=40,  # lines per chunk
    chunk_lines_overlap=15,  # lines overlap between chunks
    max_chars=1500,  # max chars per chunk
)

# %% [markdown]
# - `language`: This specifies the language of the code
# - `chunk_lines`: This defines the number of lines per chunk
# - `chunk_lines_overlap`: This defines the lines overlap between chunks
# - `max_chars`: This defines the maximum characters per chunk

# %%
nodes = splitter.get_nodes_from_documents(documents)

# %%
len(nodes)

# %%
nodes[0].to_dict()

# %%

# %%

# %% [markdown]
# ## Sentense Splitter

# %%
from llama_index.core.node_parser import SentenceSplitter

# %%
documents = SimpleDirectoryReader(input_files=['data/paul_graham_essay.txt']).load_data()

# %%
splitter = SentenceSplitter(
    chunk_size=1024, # tokens
    chunk_overlap=20, 
)

# %%
nodes = splitter.get_nodes_from_documents(documents)

# %%
len(nodes)

# %%
nodes[0].to_dict()

# %%
print(nodes[0].text)

# %%

# %% [markdown]
# ## Sentence Window Node Parser

# %%
# import nltk
from llama_index.core.node_parser import SentenceWindowNodeParser

# %% [markdown]
# - SentenceSplitter, this parser splits text into individual sentences and also includes a window of surrounding sentences in the metadata of each node. 
# - It is useful for building more context around each sentence. 
# - During the querying process, that context will be fed into the LLM and allow for better responses.

# %%
sentence_window_parser = SentenceWindowNodeParser.from_defaults(
    # how many sentences on either side to capture
    window_size=3,
    
    # the metadata key that holds the window of surrounding sentences
    window_metadata_key="window",
    
    # the metadata key that holds the original sentence
    original_text_metadata_key="original_sentence",
)

# %% [markdown]
# - `window_size`: This defines the number of sentences on each side to include in the window
# - `window_metadata_key`: This defines the metadata key for the window sentences
# - `original_text_metadata_key`: This defines the metadata key for the original sentence

# %%
nodes = sentence_window_parser.get_nodes_from_documents(documents)

# %%
len(nodes)

# %%
nodes[0].to_dict()

# %%
for i in range(10):
    print(i, nodes[i].text)
    print("-"*100)

# %%

# %% [markdown]
# ## Semantic Splitter Node Parser

# %%
from llama_index.core.node_parser import SemanticSplitterNodeParser

# %% [markdown]
# This parser requires embedding model

# %% [markdown]
# The Semantic Splitter Node Parser operates by initially dividing each sentence into segments called chunks. It then calculates the cosine dissimilarity between adjacent chunks, which measures their difference based on vector representations in semantic space. If this dissimilarity exceeds a predefined threshold, it suggests significant differences, and the chunks remain separate. Conversely, if the dissimilarity is below the threshold, indicating similarity, the chunks are concatenated into larger, unified chunks. This process helps ensure that each chunk represents a cohesive piece of information, improving the model's data processing efficiency.

# %%
from llama_index.embeddings.openai import OpenAIEmbedding

# %%
embed_model = OpenAIEmbedding()

# %%
semantic_splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model)

# %%
nodes = semantic_splitter.get_nodes_from_documents(documents)

# %%
len(nodes)

# %%
nodes[0].to_dict()

# %%
for i in range(5):
    print(i, nodes[i].text)
    print("_"*150)

# %%

# %% [markdown]
# ## TokenTextSplitter

# %%
from llama_index.core.node_parser import TokenTextSplitter

# %%
token_text_splitter = TokenTextSplitter(
    chunk_size=1024,
    chunk_overlap=20,
    separator=" ",
    backup_separators = [".", "!", "?"]
)

# %% [markdown]
# - `chunk_size`: This sets the maximum number of tokens for each chunk
# - `chunk_overlap`: This defines the overlap in tokens between consecutive chunks
# - `separator`: This is used to determine the primary token boundary
# - `backup_separators`: These can be used for additional splitting points if the primary separator doesn’t split the text sufficiently

# %%
nodes = token_text_splitter.get_nodes_from_documents(documents)

# %%
len(nodes)

# %%
nodes[0].to_dict()

# %%

# %%

# %% [markdown]
# # Relation-Based Node Parsers

# %% [markdown]
# ## Hierarchical Node Parser

# %%
from llama_index.core.node_parser import HierarchicalNodeParser

# %% [markdown]
# This parser organizes the nodes into hierarchies across multiple levels. 
# It will generate a hierarchy of nodes, starting with top-level nodes with larger section sizes, down to child nodes with smaller section sizes, where each child node has a parent node with a larger section size. 
# By default, the parser uses SentenceSplitter to chunk text. The node hierarchy looks like this:
#
# - Level 1: Section size 2,048
# - Level 2: Section size 512
# - Level 3: Section size 128
#
# The top-level nodes, with larger sections, can provide high-level summaries, while the lower nodes can allow for a more detailed analysis of text sections.
# In this way, the different node levels can be used to adjust the accuracy and depth of search results, allowing users to find information at different granularity levels. 

# %% [markdown]
# ![hierarchial node parser.jpg](attachment:65f61093-199b-41d2-bdbb-5bc43410b3ed.jpg)

# %%
hierarchical_node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])

# %%
nodes = hierarchical_node_parser.get_nodes_from_documents(documents)

# %%
len(nodes)

# %%
nodes[0].to_dict()

# %% [markdown]
# # Node parsers vs text splitters
#
# To simplify, a node parser is a more sophisticated mechanism than a simple splitter. While both serve the same basic function and operate at different levels of complexity, they differ in their implementations.
#
# - Text splitters such as SentenceSplitter can divide long flat texts into nodes, based on certain rules or limitations, such as chunk_size or chunk_overlap. The nodes could represent lines, paragraphs, or sentences, and may also include additional metadata or links to the original document.
#
# - Node parsers are more sophisticated and can involve additional data processing logic. Beyond simply dividing text into nodes, they can perform extra tasks, such as analyzing the structure of HTML or JSON files and producing nodes enriched with contextual information.
