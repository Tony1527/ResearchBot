import json
from paperqa import Docs, Settings, Doc
from paperqa.settings import ParsingSettings
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_chroma import Chroma
from pathlib import Path
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from torch import chunk
import shutil
import numpy as np
from langchain_litellm import ChatLiteLLM
import re
import hashlib
from huggingface_hub import snapshot_download

ID = str

def md5str(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.md5(data).hexdigest() 


doc_paths = ("./PaperQA2.pdf", "./PbSnSe.pdf", "./PbSnSe-copy.pdf")

CANNOT_ANSWER_PHRASE = "I cannot answer"

answer_prompt = (
    "Answer the question below with the context.\n\n"
    "Context:\n\n{context}\n\n---\n\n"
    "Question: {question}\n\n"
    "Write an answer based on the context. "
    "If the context provides insufficient information reply "
    f'"{CANNOT_ANSWER_PHRASE}." '
    "For each part of your answer, indicate which sources most support "
    "it via citation keys at the end of sentences, like \n\n"
    "{example_citation}\n\n. "
    "Only cite from the context above and only use the citation keys from the context.\n\n"
    # f"\n\n{CITATION_KEY_CONSTRAINTS}\n\n"
    "Do not concatenate citation keys, just use them as is. "
    "Write in the style of a scientific article, with concise sentences and "
    "coherent paragraphs. This answer will be used directly, "
    "so do not add any extraneous information.\n\n"
    "{prior_answer_prompt}"
)


superscript2num = {
    '⁰':"0", '¹':"1", '²':"2", '³':"3", '⁴':"4",
    '⁵':"5", '⁶':"6", '⁷':"7", '⁸':"8", '⁹':"9"
}
num2superscript = dict(zip(superscript2num.values(), superscript2num.keys()))

def get_content_from_response(res) -> str:
    if isinstance(res, dict) and res.get("messages"):
        return res["messages"][-1].content.strip()
    elif isinstance(res, list):
        return res[-1]
    else:  
        return res.content.strip()



    print("\n----------------\n")

class AnswerFormat(BaseModel):
    """Schema for answer prompt."""
    prior_answer_prompt: str | None = Field(
        default="",
        description="Prompt to include prior answer information."
    )
    context: str = Field(
        default="",
        description="The context to use for answering the question."
    )
    answer: str = Field(
        default="",
        description="The generated answer based on the context and question."
    )
    question: str = Field(
        default="",
        description="The question to answer based on the context."
    )
    example_citation: str = Field(
        default="[⁰, ¹, ², ³, ⁴, ⁵, ⁶, ⁷, ⁸, ⁹]",
        description="An example citation key to illustrate the format."
    )









class Collection():
    ''' A wrapper class for a Docs object
    '''

    def __init__(self, llm, embedding, reranker, max_sources: int=10, chunk_chars: int=500, overlap: int=50, persist_directory: str="./output/collection", **kwargs):
        print("Initializing Collection...")

        self.llm = llm
        self.embedding = embedding
        self.reranker = reranker
        self.score_threshold = 0.0

        if isinstance(self.embedding, str):
            self.embedding = HuggingFaceEmbeddings(
                model_name=embedding,
                model_kwargs={"device": "mps"}
            )

        if isinstance(self.reranker, str):
            self.reranker = CrossEncoder(reranker, device="mps")
        

        

        self.output_path = Path(persist_directory)

        # ## for testing
        # self.clear()

        if not self.output_path.exists():
            self.output_path.mkdir(parents=True, exist_ok=True)
        

        self.chunk_chars = chunk_chars
        self.overlap = overlap

        
        
        self.dockey_path = self.output_path/"dockey.json"
        if self.dockey_path.exists():
            self._dockey2file = json.load(self.dockey_path.open("r"))
            self._file2dockey = {v:k for k,v in self._dockey2file.items()}
        else:
            self._dockey2file = dict[ID, str]()
            self._file2dockey = dict[str, ID]()
        
        self.vb = Chroma(collection_name="my_personal_collection", embedding_function=self.embedding, persist_directory=str(self.output_path/"chroma_db"))
        self._doc_lock = asyncio.Lock()
        

        self.separators=[
            # 优先级最高：段落（最希望保留完整段落）
            "\n\n\n",
            "\n\n",
            # 英文/混合场景常用标点
            ". ", "! ", "? ", ".\n", "!\n", "?\n",
            # 中文常见句子结束标点（最重要！）
            "。", "！", "？", "…", 
            # 再其次：其他中文标点、半角逗号等
            "；", "; ", "，", ", ",
            # 再往后：空格、换行（避免太碎）
            " ", "\n", 
            # 最后手段：单个字符（几乎不会用到）
            ""
        ]

        self.retriever = self.vb.as_retriever(search_type="similarity",search_kwargs={"k": max_sources})

        # self.retriever = self.vb.as_retriever(search_type="mmr",search_kwargs={"k": max_sources,"fetch_k": max_sources*2,"lambda_multimodal":0.2})

        print("Finish Initializing Collection...")

    async def _process_single_doc(self, doc_path: str|Path) -> ID:
        ''' Process a single document and add it to the vector store '''
        try:
            ## 1. Load the document
            if isinstance(doc_path, str):
                path_target = Path(doc_path).resolve()
            else:
                 path_target = doc_path.resolve()  
            path_target_str = str(path_target)
            dockey = md5str(path_target.read_bytes())

            async with self._doc_lock:
                if dockey in self._dockey2file:
                    print(f"Document already exists: {path_target_str}...")
                    return dockey
                else:
                    self._dockey2file[dockey] = path_target_str
                    self._file2dockey[path_target_str] = dockey

            if path_target.suffix.lower() != ".pdf":
                print(f"Skipping non-PDF document: {path_target_str}, {path_target.suffix}...")
                return -1
            
            print(f"Processing document: {path_target.name}...")

            loader = PyPDFLoader(path_target)
            raw_doc = await asyncio.to_thread(loader.load)
            for doc in raw_doc:
                doc.page_content = doc.page_content.replace("\r", "\n")

            ## 2. Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_chars,
                chunk_overlap=self.overlap,
                separators=self.separators,
                keep_separator=False,
                strip_whitespace=True
            )


            split_chunks = text_splitter.split_documents(raw_doc)
            ids = []
            for i, chunk in enumerate(split_chunks):
                # print(chunk.metadata)
                chunk.metadata["source"] = path_target_str
                chunk.metadata["dockey"] = dockey
                ids.append(md5str(chunk.page_content+"\n"+str(i)+"\n"+path_target_str))
            
            ## 3. Add the split documents to the vector store
            if split_chunks:
                await self.vb.aadd_documents(documents=split_chunks, ids=ids)
                print(f"Successfully added {len(split_chunks)} chunks from {path_target.name}.")
            else:
                print(f"No content found in {path_target.name}.")
            return dockey
        except Exception as e:
            print(f"Error processing document {doc_path}: {e}")
            return -1

    def clear(self):
        ''' Clear the entire collection '''
        shutil.rmtree(self.output_path)
        print("Cleared the entire collection.")

    def update_dockey_file(self):
        ''' Update the dockey.json file '''
        json.dump(self._dockey2file, self.dockey_path.open("w"), indent=4)

    def is_solvable_file(self, doc_path: str|Path) -> bool:
        if isinstance(doc_path, str):
            doc = Path(doc_path)
        else:
            doc = doc_path

        return doc.suffix.lower() == ".pdf"

    async def aadd_dir(self, dir_path: str|Path):
        ''' recursively add all documents in a directory '''
        if isinstance(dir_path, str):
            cur_dir = Path(dir_path)
        else:
            cur_dir = dir_path

        all_files = [p for p in cur_dir.rglob('*') if p.is_file() and self.is_solvable_file(p)]

        print(f"Found {len(all_files)} files:")
        for file in all_files:
            print(f" - {str(file)}")

        if all_files:
            print(f"Found {len(all_files)} files in directory '{dir_path}'. Starting processing...")
            await self.aadd(all_files)
        else:
            print(f"No files found in directory '{dir_path}'.")

        

    async def aadd(self, doc_path: List[str] | str | Path):
        # Prepare the Docs object by adding a bunch of documents
        if isinstance(doc_path, str) or isinstance(doc_path, Path):
            doc_paths = [doc_path]
        elif isinstance(doc_path, list):
            doc_paths = doc_path
        else:
            raise ValueError("doc_path must be a str, Path, or list of str/Path")

        tasks = [self._process_single_doc(doc_path) for doc_path in doc_paths]
        await asyncio.gather(*tasks)
        
        self.update_dockey_file()
        

        # # 定义一个回调函数，处理流式输出
        # def on_token_callback(token: str):
        #     print(token, end="", flush=True)


    async def ainvoke(self, input: str, **kwargs):
        ''' Query the Docs object to get an answer '''
        docs = await self.retriever.ainvoke(input)

        ## score the retrived docs with cross-encoder
        scores = self.reranker.predict([[input, doc.page_content] for doc in docs ])
        scores = scores / np.linalg.norm(scores)
        

        scored_docs = []
        for doc, score in zip(docs, scores):
            doc.metadata["relevance_score"] = float(score) # 存下分数
            if score > self.score_threshold:
                scored_docs.append(doc)

        scored_docs.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)

        print(f"Score Threshold: {self.score_threshold}, Scores: {scores}")

        citations = []

        context = []
        for i, doc in enumerate(scored_docs):
            page_num = doc.metadata.get('page','Unknown')
            if page_num != 'Unknown':
                page_num = int(page_num)+1
            context.append(f"{i} [Source: {doc.metadata.get('source','Unknown')} Page:{page_num} Relevance Score:{doc.metadata.get('relevance_score','Unknown')}]:\n {doc.page_content}]")
            citations.append(f"[{i}] {doc.metadata.get('source','Unknown')}, page {page_num}")
            
        question = answer_prompt.format(
            context="\n\n".join(context),
            question=input,
            example_citation="⁽⁰, ¹, ², ³, ⁴, ⁵, ⁶, ⁷, ⁸, ⁹⁾",
            prior_answer_prompt="",
            answer=""
        )

        print(question)

        main_answer = await self.llm.ainvoke(question)
        main_content = get_content_from_response(main_answer)

        res = re.findall(r'⁽([⁰¹²³⁴⁵⁶⁷⁸⁹,\s]+)⁾', main_content)
        # print(res)
        cite_ids = []
        for group in res:
            for chars in group.strip().strip("⁽⁾").split(","):
                s = ""
                for char in chars.strip():
                    s+=superscript2num[char]
                # print(s)
                cite_ids.append(int(s))
                

        # res = re.findall(r'\[([\d,\s]+)\]', main_content)
        # # print(res)
        # cite_ids = []
        # for group in res:
        #     # group might be "0, 1" or just "0"
        #     for num_str in group.replace(" ", "").split(","):
        #         if num_str.isdigit():
        #             cite_ids.append(int(num_str))

        cite_ids = sorted(list(set(cite_ids)))

        print(cite_ids)


        citation_str = "\n".join([citations[cite_ids[i]] for i in range(len(cite_ids))])+"\n\n"

        print(main_content)
        print(citation_str)

        return main_content + citation_str







async def main():
    ''' for testing the Collection class '''
    tools = []
    streaming = True
    is_local_embedding = True
    is_local_reranker = True

    # Initialize with your desired model and enable streaming
    main_llm = ChatLiteLLM(
        model="deepseek/deepseek-chat", # Or any LiteLLM-supported model
        streaming=streaming,
        temperature=0
    )

    academic_llm= ChatLiteLLM(
        model="deepseek/deepseek-chat", # Or any LiteLLM-supported model
        streaming=False,
        temperature=0
    )

    # agent = create_agent(
    #     main_llm,
    #     tools = [parse_fuzzy_date, arxiv_search, local_files_search]
    # )






    if is_local_embedding:
        # 下载到项目目录下的 models 文件夹，从此不再依赖网络
        local_model_path = "./models/BAAI/bge-m3" 
        # local_model_path = "./models/BAAI/bge-large-en" 

        if not Path(local_model_path).exists():
            snapshot_download(
                repo_id="BAAI/bge-m3",
                local_dir=local_model_path,
                local_dir_use_symlinks=False
            )
        embedding = local_model_path

    if is_local_reranker:
        local_cross_encoder_path = "./models/cross-encoder/ms-marco-MiniLM-L-6-v2"
        if not Path(local_cross_encoder_path).exists():
            snapshot_download(
                repo_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
                local_dir=local_cross_encoder_path,
                local_dir_use_symlinks=False
            )
        reranker = local_cross_encoder_path
    col = Collection(llm=main_llm, embedding=embedding, reranker=reranker, chunk_chars=500, overlap=100, max_sources=10)
    await col.aadd_dir("./my_papers")


if __name__ == "__main__":
    asyncio.run(main())



