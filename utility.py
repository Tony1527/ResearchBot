import asyncio
import hashlib
import json
import operator
import os
import re
import shutil
import token
import warnings
import sys
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Tuple, Type, TypeVar, Union, Optional
from urllib.parse import urlencode
from importlib_metadata import metadata
from tqdm.asyncio import tqdm_asyncio

import arxiv
import numpy as np
from huggingface_hub import snapshot_download
from IPython.display import Image, display

# LangChain & LangGraph
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_litellm import ChatLiteLLM
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from langchain_core.output_parsers.json import SimpleJsonOutputParser

# Other ML/Data Libraries
from paperqa import Doc, Docs, Settings
from paperqa.settings import ParsingSettings
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
from torch import chunk


import random

async def simulate_streaming(text: str, callback=None, chunk_size=3, min_delay=0.001, max_delay=0.003):
    """
    Simulate streaming by yielding chunks of text with random small delays.
    
    Args:
        text: The complete string to stream.
        callback: Function to call with each chunk (e.g., UI update).
        chunk_size: Number of characters to send at once.
        min_delay: Minimum sleep time between chunks.
        max_delay: Maximum sleep time between chunks.
    """
    if (not callback) and (not text):
        raise ValueError("Either callback or text must be provided.")

    # 将文本切片
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        
        if callback:
            callback(chunk)
        
        # 也可以顺便打印到终端
        print(chunk, end="", flush=True)
        
        # 随机等待一小会儿，模拟打字感
        await asyncio.sleep(random.uniform(min_delay, max_delay))


def get_content_from_response(res) -> str:
    if isinstance(res, dict) and res.get("messages"):
        return res["messages"][-1].content
    elif isinstance(res, str):  
        return res
    elif isinstance(res, list):
        return res[-1]
    else:
        return res.content
    


def md5str(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.md5(data).hexdigest() 


async def stream_response(agent, inputs, streaming, callback=None, **kwargs):
    ''' Stream response from agent.
        Note: be careful that you should use await when is_stream is True
    '''
    if agent is None and isinstance(inputs, str):
        if callback is not None:
            callback(inputs)
        print(inputs)
        return inputs
    elif isinstance(agent, BaseChatModel):
        input_msg =  [HumanMessage(content=inputs)]
    else:
        if not (isinstance(inputs, dict) and inputs.get("messages")):
            input_msg = {"messages": [HumanMessage(content=inputs)]}
        else:
            input_msg = inputs

    res = ""
    if not streaming:
        response = await agent.ainvoke(input=input_msg, **kwargs)
        res += get_content_from_response(response)
        if callback:
            callback(res)
        print(res)


        # for token, metadata  in agent.stream(input=input_msg,stream_mode="messages", **kwargs):
        #     if token.content:
        #         print(token.content, end="", flush=True)
    else:
        async for tmp in agent.astream(input=input_msg,stream_mode="messages", **kwargs):
            if isinstance(tmp, tuple) and len(tmp) == 2:
                token, metadata = tmp
            else:
                token = tmp
            if token.content:
                if callback is not None:
                    callback(token.content)
                print(token.content, end="", flush=True)
                res += token.content

    print("\n----------------\n")
    return res



def create_schema_output(llm, base_model_name: BaseModel, input: str) -> Dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Output ONLY valid JSON. No explanation, no markdown, no fences, no extra text.
            Task: Extract key information from the following and structure it in JSON.
            Required JSON schema:
            {schema}
            """),
        ("human", "{input}")
    ])

    # schema_str = json.dumps(CustomData.model_json_schema(), indent=2)
    # prompt = prompt.partial(schema=schema_str)
    
    prompt = prompt.partial(schema = base_model_name.model_json_schema())
    # print(prompt)

    # LCEL 链
    chain = prompt | llm | SimpleJsonOutputParser()
    res = chain.invoke({"input": input})
    return res


def create_schema_by_tool_binding(llm, base_model_name: BaseModel, input: str) -> BaseModel:
    chain = llm.bind_tools([base_model_name])
    res = chain.invoke(input)
    print(input)
    print(res)
    return base_model_name.model_validate(res.tool_calls[-1]["args"])
