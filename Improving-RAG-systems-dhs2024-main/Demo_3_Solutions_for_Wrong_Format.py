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
# <a href="https://colab.research.google.com/github/dipanjanS/improving-RAG-systems-dhs2024/blob/main/Demo_3_Solutions_for_Wrong_Format.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="b8kkhS-UgJAR"
# # Solutions for Wrong Format
#
# Here we will explore the following strategies
#
# - Native LLM Support
# - Output Parsers
#

# %% [markdown] id="L1KvMtf54l0d"
# #### Install OpenAI, HuggingFace and LangChain dependencies

# %% id="2evPp14fy258" colab={"base_uri": "https://localhost:8080/"} outputId="4d51f351-9919-4245-fd5f-f07551588b2a"
# !pip install langchain
# !pip install langchain-openai
# !pip install langchain-community

# %% [markdown] id="-E6SF7YdioWZ"
# ### Enter Open AI API Tokens

# %% colab={"base_uri": "https://localhost:8080/"} id="eeBpx1YEioWg" outputId="32d8fb54-d19b-4b91-d807-5a9e19781f11"
from getpass import getpass

OPENAI_KEY = getpass('Enter Open AI API Key: ')

# %% id="x1YSuHNF_lbh"
import os

os.environ['OPENAI_API_KEY'] = OPENAI_KEY

# %% [markdown] id="yHhc7UKVX7nC"
# # Native LLM Output Response Support

# %% id="9qjM8sSzCfg2"
from langchain_openai import ChatOpenAI

chatgpt = ChatOpenAI(model_name="gpt-4o-mini", temperature=0,
                     model_kwargs={"response_format": {"type": "json_object"}})

# %% id="WTHZDxdPCfcP"
prompt = """Who won the Champions league in 2023,
            Output should be in JSON and have following fields:
            win_team, lose_team, venue, date, score
         """
response = chatgpt.invoke(prompt)

# %% colab={"base_uri": "https://localhost:8080/"} id="-5nr2eBgCfZr" outputId="d3fe2337-cf54-400f-e8d3-9bb023b0ca14"
print(response.content)

# %% colab={"base_uri": "https://localhost:8080/"} id="xPEFVAUCDqGV" outputId="1308059a-cbde-4d34-ed9a-96b75fbc46a6"
type(response.content)

# %% [markdown] id="zBGV2SazYBIL"
# # Output Parsers

# %% id="Td0hVmQd3OVw"
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


# Define your desired data structure - like a python data class.
class GameDetails(BaseModel):
    win_team: str = Field(description="The winning team in the football game")
    lose_team: str = Field(description="The losing team in the football game")
    venue: str = Field(description="The venue of the football game")
    date: str = Field(description="The date of the football game")
    score: str = Field(description="The score of the football game")

parser = JsonOutputParser(pydantic_object=GameDetails)

# %% id="xFIk3rqn4RLS" colab={"base_uri": "https://localhost:8080/"} outputId="415ea1f0-aed9-4e60-c074-f66a48ad5e4a"
print(parser.get_format_instructions())

# %% id="1HJvSevb4XES"
from langchain_core.prompts import PromptTemplate

prompt_txt = """
             Who won the Champions league in 2023
             Use the following format when generating the output response

             Output format instructions:
             {format_instructions}`
             """

prompt = PromptTemplate.from_template(template=prompt_txt)

# %% id="ObjYJrXh8Hj3"
llm_chain = (prompt
              |
            chatgpt
              |
            parser)

response = llm_chain.invoke({"format_instructions": parser.get_format_instructions()})

# %% colab={"base_uri": "https://localhost:8080/"} id="rWdKHqCIF5qO" outputId="12b4ea4c-01d4-4cc5-c2ec-fa2bb1a1df90"
response

# %% colab={"base_uri": "https://localhost:8080/"} id="P2x98vgfF62K" outputId="5681c383-5c6d-4d0e-887e-3de644bf8a77"
type(response)
