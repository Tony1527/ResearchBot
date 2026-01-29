import asyncio
import hashlib
import json
import operator
import os
import re
import shutil
import warnings
from datetime import date
from pathlib import Path
from typing import Annotated, List, Literal, Optional
from urllib.parse import urlencode

import arxiv
import numpy as np
from huggingface_hub import snapshot_download
from IPython.display import Image, display

# LangChain & LangGraph
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_litellm import ChatLiteLLM
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

# Other ML/Data Libraries
from paperqa import Doc, Docs, Settings
from paperqa.settings import ParsingSettings
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
from torch import chunk


def get_content_from_response(res) -> str:
    if isinstance(res, dict) and res.get("messages"):
        return res["messages"][-1].content.strip()
    elif isinstance(res, list):
        return res[-1]
    else:  
        return res.content.strip()


def md5str(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.md5(data).hexdigest() 


async def stream_response(agent, inputs, streaming, **kwargs):
    ''' Stream response from agent.
        Note: be careful that you should use await when is_stream is True
    '''
    if not (isinstance(inputs, dict) and inputs.get("messages")):
        input_msg = {"messages": inputs}

    if not streaming:
        response = await agent.ainvoke(input=input_msg, **kwargs)
        print(get_content_from_response(response))


        # for token, metadata  in agent.stream(input=input_msg,stream_mode="messages", **kwargs):
        #     if token.content:
        #         print(token.content, end="", flush=True)
    else:
        async for token, metadata  in agent.astream(input=input_msg,stream_mode="messages", **kwargs):
            if token.content:
                print(token.content, end="", flush=True)

    print("\n----------------\n")
