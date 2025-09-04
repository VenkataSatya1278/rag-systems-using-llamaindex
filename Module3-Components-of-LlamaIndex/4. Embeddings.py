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

# %% [markdown] id="4EVx6wb6hFk8"
# # Embeddings
#
# In this notebook, we will explore how to access different types of embeddings
# in llamaindex
#
#
# 1.   OpenAI
# 2.   Google Gemini
# 3.   CohereAI
# 4.   Open-Source from HuggingFace
#
#

# %% [markdown]
# Download the required packages by executing the below commands in either Anaconda Prompt (in Windows) or Terminal (in Linux or Mac OS)

# %% [markdown]
# pip install llama-index-embeddings-gemini llama-index-embeddings-cohere

# %% [markdown]
# ## Content
# - Documents -> Chunks -> Embeddings -> index
# - Why do we need Embeddings?
# - Different embedings for different openai companies
# - Types of Embeddings -> word based, sentence based, document based,
# - What kind of embeddings the LLM will use?
# - Flow diagram/ Architecture representation of Embeddings
# - Interpreting Embeddings -> Cosne Similarity formula to determine the similarilty in two vectors regarding of their magnitude.
# - Applications of Embeeddings - How o find the most similar words, finding the odd one , document clustering for different types of embeddings.
# - Closed source Embeddings - which are paid embeddings
# - Open source Embeddings 
# - How to select the right embeddings for my use case?
# - Massive Text Embedding Benchmark(MTEB) - To prove the model is better than other model. Check Top Embedding from MTEB LearderBoard. https://huggingface.co/spaces/mteb/leaderboard 
# - Pre-trainned vs Fine Tunned embeddings- 

# %%
# !pip install dotenv
# !pip install llama-index-embeddings-gemini llama-index-embeddings-cohere 
# !pip install llama_index.embeddings.openai

# %% [markdown]
# ## Load the Keys

# %%
import os
from dotenv import load_dotenv, find_dotenv

# %%
load_dotenv('D:/Training/FAA-Training/Beyond-the-Prompt-Practical-RAG-for-Real-World-AI/RAG-systems-using-LlamaIndex/Module3-Components-of-LlamaIndex/.env')

# %%
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
HUGGINGFACE_API_KEY = os.environ['HUGGINGFACE_API_KEY']
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

# %% [markdown] id="SQ8e5Zlan50w"
# ## 1. OpenAI Embeddings

# %% id="2MQ52KlvhLkA"
from llama_index.embeddings.openai import OpenAIEmbedding

# %% id="VUdomDpZiFLS"
# Define embedding model with OpenAI Embedding
embed_model = OpenAIEmbedding(model='text-embedding-3-small', api_key=OPENAI_API_KEY)

# %% id="ZlFLxcgzmbEQ"
# Get the text embedding
embedding = embed_model.get_text_embedding("The cat sat on the mat")

# %% colab={"base_uri": "https://localhost:8080/"} id="JlM3B4aViReS" outputId="c06e6c1d-6968-4f89-c234-d0a3b9df9225"
# Get the dimension of the embedding
len(embedding)

# %%
embedding[:10] 

# %% id="p9q7AgtCiUEW"
# You can get embeddings in batches
embeddings = embed_model.get_text_embedding_batch(["What are Embeddings?", \
                                                   "In 1967, a professor at MIT built the first ever NLP program Eliza to understand natural language."])

# %% colab={"base_uri": "https://localhost:8080/"} id="-dIvFjYWihyQ" outputId="07e47786-010d-474a-a225-185048d416d1"
len(embeddings)

# %%
embeddings[0][:5] # embeddings of the 1st sentence

# %% colab={"base_uri": "https://localhost:8080/"} id="Fqr9wDhcijz_" outputId="bcf64c72-d299-4c6e-9762-2966272d309e"
embeddings[1][:5]  # embeddings of the 2nd sentence

# %%
len(embeddings[1]) # length of each embedding

# %% [markdown] id="pStpFtd4iTJe"
# ## 2. Using Google Gemini Embeddings

# %% id="rwwi9ihoiR35"
# imports
from llama_index.embeddings.gemini import GeminiEmbedding

# %% id="53lb4RDsiXbZ"
model_name = "models/text-embedding-004"

# %% id="FSlhNj2-iXeJ"
embed_model = GeminiEmbedding(model_name=model_name, api_key=GOOGLE_API_KEY)

# %% id="2mYeQ0AoiXgT"
embeddings = embed_model.get_text_embedding("A journey to the centre of Earth")

# %% colab={"base_uri": "https://localhost:8080/"} id="aG88KEnzihpH" outputId="cb92a104-1212-4d71-a8a6-dbb810e2f318"
print(f"Dimension of embeddings: {len(embeddings)}")

# %% colab={"base_uri": "https://localhost:8080/"} id="9ie4cYIsilAX" outputId="10649990-c6e9-46ba-f46d-998d06b1c8af"
embeddings[:5]

# %% [markdown] id="DXkBrFjzjRpx"
# ## 3. Using CohereAI Embeddings

# %% id="uW3OYf7Pja6r"
from llama_index.embeddings.cohere import CohereEmbedding

# %%
embed_model = CohereEmbedding(
    cohere_api_key=COHERE_API_KEY,
    model_name="embed-english-v3.0",
    input_type="search_query",
)

# %% id="-5GNNFamje23"
embeddings = embed_model.get_text_embedding("Hello CohereAI!")

# %% colab={"base_uri": "https://localhost:8080/"} id="ezqGGSvrjgen" outputId="f95612af-0f1a-4ed2-dbb0-0fba3a3f979a"
print(len(embeddings))

# %% colab={"base_uri": "https://localhost:8080/"} id="kx8g0LYijghz" outputId="aa7e1034-ab2c-4617-f9f9-6319deba8234"
print(embeddings[:5])

# %% [markdown] id="YuAcq_Ufn-As"
# ## 4. Open Source Embeddings from HuggingFace.

# %% colab={"base_uri": "https://localhost:8080/", "height": 209, "referenced_widgets": ["18720ba4e702464cac6e283a19fcc1ef", "2939870d643a471b937e2bdf5e8350db", "6550569f2cb84af2b1a6afe6423c5606", "93160150167b426bb8ab855884a95d2d", "36c340ea8b264c53b42fb548c04db863", "91f7ef1bd909445b91cb9a0ba6218547", "2158a9d86b8048b4a3ed1d958be18c95", "93020ac5c9fa403a9558e3e1c9ddb54d", "ba399ab40e4949298c3567ad7f533f08", "866d1b36545d46d28c36122b11cb6775", "9ba2df42cd7c428ab9c8149792d0d8ac", "ec8ac82c03f14705863e49851e430e62", "ca671524043f4a49b15725fa796fa54c", "3c01a086e53e47129be2acb4b0737906", "c302447a43c144fc93aa0cbb4fbc9aa9", "0610fbf0fc0d4fd2b1866a1c7fd95c6e", "a5e2f90b65574419a1e115e0826081b2", "f65f2c1a84a5407684e837eb73b13a57", "b09d9ec930664352a3b64ad9ad850692", "12426a58b18943b69de081dec08036d2", "531ea44bf75443bc8c8908c750efd692", "353f78df62c84ad28b9dcfdaafa69d1a", "d99c5e1516d64b59b3f84b4be4d4114b", "c16642d677b545dd9e4bf9d21b45d703", "b531ca9c7e974ab883358626e549ddd2", "9b443dbdbaad4ccfa89f84bb8c243040", "c35b0722500d449994ac53244ce59d8b", "2fcdf9c20d074c4da7a7a0cac238fb14", "9aa953206bb54ff7af3a7b30ed78bfde", "20228d8fe68c41d2b867260dfd059c56", "e6d758b39d394653bf8bc2aaac722b1d", "3221d50bc7ed4d7d89122d3db3198047", "ca9b35fe04e0493e905174dd0f0c8769", "3f6d81fd0a624a85a4d09ce05ea2d03c", "88dc39714c5f4e20a935238996133f91", "04b20460ea984380b283fb7d5d275cc7", "6ee629c76fa94bae9c5428a01b9086ba", "a2f9a4f7e9be41f1b11faa2eaa30c23c", "c7fc7c77fca54ad193bd26388017f779", "fc2584a4ac134535ad7775f91f673cdf", "e0a53f59761c418583e0afea0b7b247a", "1172cea981b74709bdc42cde62d43105", "07d95aa24f72439f81108682b1731407", "6f119b9aefbd44dcaa8c620d54118b45", "4a372900d5e840eda99c0fcaa7b0e8d3", "a78a2421da8c4f96b87117d1b8ec1d40", "e5b1aca039da407e98ed723d8e93ce50", "57e0b60c76fc460ea882d89dfeac5823", "1f646d8769374edd8443c9ba3bfb843e", "5c0f3cd9a1d74d73b5a1b81725a1c9e5", "acfcaf3b51b8492fa8a6f47e44ebb4c3", "980a3a2d1aa54933a82d1ac33df52feb", "b4ee6a2f2b514a11b21346f9fccb042c", "68936a6880f14cfcbf406df5697cd9c0", "9996baee969d4de59d58d4f83cbd6b1a", "f834043a0fa749eeb9a4741de299e520", "22e5ad1815b84d0782f66c07faec2fd8", "bfb0cf35c8a54bb1acd56023b73b69cf", "e67a74af17fe43e09be6a3245b921f33", "a9a66e8446a041f2917c05cec3e4bfcf", "b65d9ed6830446b18155ed57da709e80", "1de2c8bcac5e47d0b1153efccd8cf5f5", "e97fa2e309134e1bb1a714b0137c5846", "a7ffdcd51c8745f89c0a1e2983bb0898", "acd1a39cff8349658ca1f5cfe9b395e1", "3d3a86a6c03a40e086fe756b08ab22e5"]} id="nC4jOfL4imRs" outputId="bc8b101d-fc29-4824-a738-23967317e633"
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# %%
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# %% id="-mUAXhoZljyj"
embedding = embed_model.get_text_embedding("Roses are red, and the sky is blue!?")

# %% colab={"base_uri": "https://localhost:8080/"} id="y_DJ697xlu-Z" outputId="84c66cb2-1f39-459c-b54a-8e485ce60705"
len(embedding)

# %% id="FXsEVehImDQM"
embeddings = embed_model.get_text_embedding_batch(["Hugging Face Text Embeddings Inference",
                                                   "OpenAI Embedding",
                                                   "Open Source",
                                                   "Closed Source"])

# %% colab={"base_uri": "https://localhost:8080/"} id="jKOl3IkxmR7j" outputId="d1db8dc2-8bf2-4c70-bf13-e7fb8167e1f6"
len(embeddings)

# %% colab={"base_uri": "https://localhost:8080/"} id="aUJe6-7smUbX" outputId="8ed22b08-5ee5-4355-d6af-185e9c1456c8"
embeddings[0][:5]

# %%
len(embeddings[0])

# %% [markdown] id="tUsPQV0QknKV"
# ## 5. Loading SOTA Embedding Model

# %% colab={"base_uri": "https://localhost:8080/", "height": 209, "referenced_widgets": ["a387c669042a47a084695476a32865ed", "5fa8f1435b7844448233e192c8bcb78e", "8fab0a50768d409eab7dc096b3fc32a9", "ad1ec2d9c0dc4f88a5b37c4c53b9cc79", "ee4c1e1dc72249459a391e5b23d1c1a9", "a4690ef9a745442d9ea53353239bf898", "fbc12ad7c42248789e5f16412da45cc9", "56f3be97398641f792527163052bca78", "2fc83eb3f52d49a0a7a9a5cb3f2aeb55", "f690a90b1e44439b8e30d3152f2d3043", "9d85dc82cfc445f69a39e6997493800e", "3b686a87765c4508bc097a5603666fe8", "d0dd34b8b16b4738bf3823ef99fa62c7", "98bcbad994464a1e90c8cea504043d59", "c620b878ae6c4fa5a170c9d64567d5a2", "a9bdab38689445a7be934ae7e996f84b", "693f5d73f07d4fcb9de3689040efddfb", "aa339739f4544e57a0f057aed992eb0f", "218dfa83d09a4dd2841e5633c06d3075", "83600a55fd5941619bd4580c8ed5c295", "21556edf6c9746d791ee855c2ea43347", "174d75394d0642d5baee7c918a7676a5", "f18b9284633247daa57be1e4787ae570", "569c2f8072544c598a50ed225319d509", "c4e8dcc2db674b8ea084e39f856b6604", "2531c0c0b5044992ab6fb96c8110dbe2", "04506652139b4cfd8e36670f301fc112", "725fe9f0dee649ed98eb33501e910568", "d6297b23e40d4d88b76f9fc29f11091d", "72d6490cc8ed41cca8ade992c5c8ed88", "c11f01032687435fb2847b89aa7b1548", "ff3351be1c4f4f9ab0cf6b62b23c3626", "a316b9ee16b9431b9a56ae3d2af5cf61", "1c09e0d16a5e4a5ca50584f3cf0e3257", "021a32e8926244f392d9e45ea9da7514", "06c5e205fc7f4acfb6d857b01330a172", "930968efda3b4c50b16b0b915401f01f", "0d58a21c80254413ba74facddf2629df", "70ae0df1683840e7ab8dba0832e182d4", "6a7126140f054769a97a2f58c41adfba", "bba85c87a69840c6bf1b693d9c24b500", "fe9799d8dbb54b6eb81e2769a65be499", "b66e34478ec449c6b70ef57e427e76e6", "88ab99cde32c4d00b7b7ea71209772a1", "0f1749b1571e4af0a0ef326e5d53be47", "5398943eb06343fe9110e15ed5ac4cca", "c6da76b8ea5d4e72a405f9fa7ef05989", "09d31bee24224982a1686e7c1430cb5a", "106e3860db574788b725aa621022e742", "03de7f2564e34c578ca978dc1d7b281f", "7a255b02561f44a5a7a9cd2c9b7aefd0", "b0ecdb6344274770960ac88a62656902", "6648d95456b943a2852a9c2ba70025e2", "d0d06ffc74ed48bd902229528f8596e5", "d8901b9333bf4ebe9af8498deb5e8a9e", "e665360b0e7848f3a0f71ad1275bf473", "00a2380cf9fe409481c16cf00f67c5ac", "0a741c03df8a4787ad1a72d22cd98b4d", "40702a56f5a54a0f82d509887d6271bd", "f9344c50b26540a88845e1c0eaca1b13", "9c50d96e35fc44e6958c57e96d0f8198", "3cac8cd0a780413dbb5d5da9f9c0d3ad", "a048deb17f1f4e2d9b0864596de97906", "3b46322c302340919457c2e8f42ff5a1", "5fdba18b92854016890d0cc36d0257a1", "a06e8a0991894579999563dd7fa2db75"]} id="8VlOHm49kOKj" outputId="85e4b031-2809-4603-eddd-57692a837570"
embed_model = HuggingFaceEmbedding(model_name='WhereIsAI/UAE-Large-V1')

# %% id="G-PYCP_iktfE"
embedding = embed_model.get_text_embedding("Hugging Face Text Embeddings Inference")

# %% colab={"base_uri": "https://localhost:8080/"} id="QWUF-kc6kyOT" outputId="e0a89c9e-1a0e-4cf6-c6d1-5351a791979b"
print(embedding[:5])

# %% colab={"base_uri": "https://localhost:8080/"} id="vaxDYpCqkthN" outputId="b3e486a6-db9d-4596-fc5f-5d33d9106dcd"
len(embedding)

# %% [markdown] id="oblsMNndk9y8"
# ## You can check the top embeddings from this leaderboard: https://huggingface.co/spaces/mteb/leaderboard
