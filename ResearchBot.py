from typing import Callable
from litellm import embedding
from sentence_transformers import SentenceTransformer
from DeepLearning.ResearchBot import tools
from prompt import *
from utility import *
from tools import *

from agentmail import AgentMail
from langchain_core.messages import BaseMessage, AnyMessage
from langgraph.graph import MessagesState
import time

## remove the warning from pydantic when using the LiteLLM 
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Pydantic serializer warnings",
    module="pydantic.main"
)

ID = str




superscript2num = {
    '⁰':"0", '¹':"1", '²':"2", '³':"3", '⁴':"4",
    '⁵':"5", '⁶':"6", '⁷':"7", '⁸':"8", '⁹':"9"
}
num2superscript = dict(zip(superscript2num.values(), superscript2num.keys()))

class AnswerFormat(BaseModel):
    '''Schema for answer prompt.
    '''
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
        default="⁽⁰, ¹, ², ³, ⁴, ⁵, ⁶, ⁷, ⁸, ⁹⁾",
        description="An example citation key to illustrate the format."
    )

class EmailFormat(BaseModel):
    ''' Schema for sending an email.
    '''
    to_addr: str = Field(..., description="The recipient's email address.")
    subject: str = Field(..., description="The subject of the email.")
    text: str = Field(..., description="The body text of the email.")
    from_addr: Optional[str] | None = Field(None, description="The sender's email address. If None, use the default inbox address.")

class ArxivSearch(BaseModel):
  ''' Schema for searching arxiv papers.
  '''
  keywords: Optional[List[str]] | None = Field(None, description="Scientific terms, model names, topics to search.")
  must_have_all_keywords: bool = Field(True, description="Whether all keywords must be present in the search results.")
  categories: Optional[List[str]] | None = Field(None, description="arXiv categories like 'cond-mat', 'quant-ph' etc. By default, it should search 'cond-mat' and 'quant-ph'.")
  must_have_all_categories: bool = Field(True, description="Whether all categories must be present in the search results.")
  authors: Optional[List[str]] | None = Field(None, description="List of authors to search for. Author names in 'Firstname Lastname' format.")
  must_have_all_authors: bool = Field(True, description="Whether all authors must be present in the search results.")
  date_from: Optional[date] | None = Field(None, description="Start date in 'YYYY-MM-DD' format. Example: '2023-01-01'.")
  date_to: Optional[date] | None = Field(None, description="End date in 'YYYY-MM-DD' format. Defaults to today.")
  max_results: Optional[int] | None = Field(50, description="Maximum number of results to return")
  sort_by: Optional[Literal["relevance", "lastUpdatedDate", "submittedDate"]] | None = Field("submittedDate", description="Criterion to sort results by")
  markdown_format: bool = Field(False, description="Whether to format the results in markdown. By default, it should be False.")

class ArxivResult(BaseModel):
    ''' Schema for a single arxiv paper result.
    '''
    title: str = Field("", description="Title of the paper.")
    authors: List[str] = Field(default_factory=list, description="List of authors of the paper.")
    full_summary: str = Field("", description="Full summary of the paper.")
    pdf_url: str = Field("", description="URL to download the PDF of the paper.")

    updated: str = Field("", description="The date when the paper was last updated.")
    published: str = Field("", description="The date when the paper was published.")
    doi: Optional[str] = Field(None, description="DOI of the paper, if available.")

    refined_summary: str = Field("", description="A concise summary of the paper's key contributions and findings.")
    score: int = Field(0, description="Relevance score of the paper from 0 to 100 based on the search criteria.", ge=0, le=100)

class ArxivResults(BaseModel):
    ''' Schema to combine all arxiv search results.
    '''
    query: str = Field("", description="Query used for the search.")
    total_results: int = Field(0, description="Total number of results found.")
    papers: List[ArxivResult] = Field(default_factory=list, description="List of papers found.")


class CustumKeywords(BaseModel):
    keywords: List[str] = Field(..., description="List of custom keywords for searching.")

class User(BaseModel):
    id: ID = Field(..., description="Unique identifier for the user.")
    name: str = Field(..., description="Name of the user.")
    email: str = Field(..., description="Email address of the user.")
    custom_keywords: Optional[CustumKeywords] | None = Field(None, description="Custom keywords associated with the user.")


class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    users: Dict[ID, User] = Field(default_factory=dict, description="List of users involved in the conversation.")
    username2id: Dict[str, ID] = Field(default_factory=dict, description="Mapping from username to user ID.")

class RouterChoice(BaseModel):
    reasoning: str = Field(
        "Step-by-step reasoning behind the classification"
    )
    classification: Literal["do_research_event", "send_email_event"] = Field(
        description="The classification of user's input:"
        "'do_research_event' for searching the arxiv and local files; "
        "'send_email_event' for sending emails."
    )

class SendEmailDetails(BaseModel):
    to_addr: str = Field(..., description="Recipient email address")
    content_description: str = Field(
        ..., 
        description="Description of what content to include in the email (e.g., 'recent arxiv papers about graphene')"
    )
    requires_search: bool = Field(
        False, 
        description="True if the email content requires searching external sources first"
    )


class ArxivRefine(BaseModel):
    keywords: List[str] = Field(..., description="Three keywords based on the title and abstract.")
    refined_summary: str = Field(
        ..., 
        description="The core contribution and methodology in one concise sentence based on the title and abstract."
    )



def convert_date_to_arxiv_format(d: date) -> str:
    ''' Convert a date object to the arxiv date format 'YYYYMMDDHHmm'. '''
    return "".join(d.isoformat().split("-"))+"0600"

async def summarize_abstract(chain, title, full_summary) -> str:
    ''' Summarize the abstract using the provided chain.'''
    if not full_summary:
        return ""
    try:
        res = await chain.ainvoke({"title": title, "full_summary": full_summary})
        return get_content_from_response(res)
    except Exception as e:
        print(f"Error in summarize_abstract: {e}")
        return ""

def get_device():
    os_name = sys.platform
    if os_name == "darwin":
        device = "mps"
    else:
        device = "cuda"
    

    return device


def get_embedding_and_cross_encoder(embedding_model, cross_encoder_model, device, is_local_embedding=True, is_local_cross_encoder=True):
    if isinstance(embedding_model,str):
        embedding_instance = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device, "local_files_only": is_local_embedding}
        ) if isinstance(embedding_model, str) else embedding_model
    else:
        embedding_instance = embedding_model

    if isinstance(cross_encoder_model, str):
        cross_encoder_instance = CrossEncoder(cross_encoder_model, device=device, local_files_only=is_local_cross_encoder) if isinstance(cross_encoder_model, str) else cross_encoder_model
    else:
        cross_encoder_instance = cross_encoder_model

    return embedding_instance, cross_encoder_instance


class Collection():
    ''' A wrapper class for a Docs object
    '''

    def __init__(self, llm, embedding, reranker, max_sources: int=10, chunk_chars: int=500, overlap: int=50, persist_directory: str="./output/collection", streaming=True,**kwargs):
        print("Initializing Collection...")

        self.llm = llm
        self.embedding = embedding
        self.reranker = reranker
        self.score_threshold = 0.0
        self.streaming = streaming


        self.embedding, self.reranker = get_embedding_and_cross_encoder(embedding, reranker, get_device())
        self.callback = None

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
            # Priority highest: paragraphs (prefer to keep complete paragraphs)
            "\n\n\n",
            "\n\n",
            # Common punctuation used in English/mixed contexts
            ". ", "! ", "? ", ".\n", "!\n", "?\n",
            # Common Chinese sentence-ending punctuation (most important)
            "。", "！", "？", "…", 
            # Next: other Chinese punctuation and half-width commas
            "；", "; ", "，", ", ",
            # Then: spaces and newlines (avoid creating very small chunks)
            " ", "\n", 
            # Last resort: single character (rarely used)
            ""
        ]

        self.retriever = self.vb.as_retriever(search_type="similarity",search_kwargs={"k": max_sources})

        # self.retriever = self.vb.as_retriever(search_type="mmr",search_kwargs={"k": max_sources,"fetch_k": max_sources*2,"lambda_multimodal":0.2})

        print("Finish Initializing Collection...")

    def set_callback(self, callback):
        ''' Set the callback function for streaming responses '''
        self.callback = callback

    async def _process_single_doc(self, doc_path: str|Path, semaphore: asyncio.Semaphore) -> ID:
        ''' Process a single document and add it to the vector store '''
        async with semaphore:
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

                if path_target.suffix.lower() != ".pdf":
                    print(f"Skipping non-PDF document: {path_target_str}, {path_target.suffix}...")
                    return -1
                
                print(f"Processing document: {path_target.name}...")

                # loader = PyPDFLoader(path_target)
                loader = PyMuPDFLoader(path_target)
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

                    async with self._doc_lock:
                        if dockey in self._dockey2file:
                            print(f"Document already exists: {path_target_str}...")
                            return dockey
                        else:
                            self._dockey2file[dockey] = path_target_str
                            self._file2dockey[path_target_str] = dockey
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
        ## @deprecated
        if isinstance(dir_path, str):
            target_dir = Path(dir_path)
        else:
            target_dir = dir_path

        all_files = [p for p in target_dir.rglob('*') if p.is_file() and self.is_solvable_file(p)]
        print(f"Found {len(all_files)} files:")
        # for file in all_files:
        #     print(f" - {str(file)}")

        if all_files:
            print(f"Found {len(all_files)} files in directory '{dir_path}'. Starting processing...")
            await self.aadd(all_files)
        else:
            print(f"No files found in directory '{dir_path}'.")

    async def aadd(self, path: List[str] | str | Path):
        # Prepare the Docs object by adding a bunch of documents
        if isinstance(path, str) or isinstance(path, Path):
            doc_paths = [path]
        elif isinstance(path, list):
            doc_paths = path
        else:
            raise ValueError("doc_path must be a str, Path, or list of str/Path")
        
        all_files = []
        for doc_obj in doc_paths:
            if not isinstance(doc_obj,Path):
                doc_path = Path(doc_obj)
            else:
                doc_path = doc_obj

            print(f"doc_path.is_dir(): {doc_path.is_dir()}")
            print(f"doc_path.is_file(): {doc_path.is_file()}")

            if doc_path.is_dir():
                all_files.extend([p for p in doc_path.rglob('*') if p.is_file() and self.is_solvable_file(p)])
            elif doc_path.is_file():
                all_files.append(doc_path)
            else:
                raise ValueError("Unsupported type.")

        if all_files:
            print(f"Found {len(all_files)} files.")
        else:
            print(f"No files found.")


        semaphore = asyncio.Semaphore(4)  # Limit to 4 concurrent tasks

        tasks = [self._process_single_doc(file_path, semaphore) for file_path in all_files]
        await tqdm_asyncio.gather(*tasks)
        
        self.update_dockey_file()
        
    async def ainvoke(self, input: str, **kwargs):
        ''' Query the Docs object to get an answer '''
        docs = await self.retriever.ainvoke(input)

        ## score the retrived docs with cross-encoder
        scores = self.reranker.predict([[input, doc.page_content] for doc in docs ])
        scores = scores / np.linalg.norm(scores)
        

        scored_docs = []
        for doc, score in zip(docs, scores):
            doc.metadata["relevance_score"] = float(score) # save the score
            if score > self.score_threshold:
                scored_docs.append(doc)

        scored_docs.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)

        await simulate_streaming("\n\n", callback=self.callback) ## for testing streaming

        print(f"Score Threshold: {self.score_threshold}, Scores: {scores}")

        citations = []

        context = []
        for i, doc in enumerate(scored_docs):
            page_num = doc.metadata.get('page','Unknown')
            if page_num != 'Unknown':
                page_num = int(page_num)+1
            context.append(f"{i} [Source: {doc.metadata.get('source','Unknown')} Page:{page_num} Relevance Score:{doc.metadata.get('relevance_score','Unknown')}]:\n {doc.page_content}]")
            citations.append(f"[{i}] {doc.metadata.get('source','Unknown')}, page {page_num}, '{doc.page_content[:30]}...'")
        
        default_answer_format = AnswerFormat()

        question = answer_prompt.format(
            context="\n\n".join(context),
            question=input,
            example_citation=default_answer_format.example_citation,
            prior_answer_prompt=default_answer_format.prior_answer_prompt,
            answer=default_answer_format.answer
        )

        print(question)

        # main_content = await stream_response(self.llm, inputs=question, streaming=False, callback=self.callback)

        main_content = await self.llm.ainvoke(question)
        main_content = get_content_from_response(main_content)

        # main_content = await stream_response(self.llm, inputs=question, streaming=self.streaming, callback=self.callback)


        # main_answer = await self.llm.ainvoke(question)
        # main_content = get_content_from_response(main_answer)

        res = re.findall(r'[⁽\[\()]([⁰¹²³⁴⁵⁶⁷⁸⁹,\s\d]+)[⁾\]\)]', main_content)
        # print(res)
        cite_ids = []
        for group in res:
            for chars in group.strip().strip("⁽⁾").split(","):
                s = ""
                for char in chars.strip():
                    s+=superscript2num[char]
                # print(s)
                cite_ids.append(int(s))
            

        cite_ids = sorted(list(set(cite_ids)))

        # print(cite_ids)


        citation_str = "\n".join([citations[cite_ids[i]] for i in range(len(cite_ids))])+"\n\n"
        reference_str = "\n\nReference:\n"  + citation_str

        res = main_content + reference_str
        await simulate_streaming(text=reference_str, callback=self.callback)
        


        return res







    

class ResearchBotConfig(BaseModel):
    ''' Configuration schema for ResearchBot
    '''
    is_local_llm: bool = Field(False, description="Whether to use local LLM model.")
    is_local_embedding: bool = Field(True, description="Whether to use local embedding model.")
    is_local_cross_encoder: bool = Field(True, description="Whether to use local cross encoder.")
    llm_model: str = Field("deepseek/deepseek-chat", description="LLM model path or API.")
    llm_api_key: Optional[str] = Field(None, description="API key for the LLM if required.")
    llm_api_base: Optional[str] = Field(None, description="Base URL for the LLM API if required.")
    embedding_model: str = Field("BAAI/bge-m3", description="Embedding model path or API.")
    cross_encoder_model: str = Field("BAAI/bge-reranker-base", description="Cross-encoder's path or API.")
    # cross_encoder_model: str = Field("cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder's path or API.")

    system_prompt: str = Field(research_agent_prompt, description="System prompt for the agent.")

    max_sources: int = Field(10, description="Maximum number of sources to retrieve.")
    chunk_chars: int = Field(500, description="Number of characters per document chunk.")
    overlap: int = Field(100, description="Number of overlapping characters between chunks.")
    persist_directory: str = Field("./output/collection", description="Directory to persist the collection data.")
    local_files_dir: str = Field(".", description="Directory path for local files to search.")
    streaming: bool = Field(True, description="Whether to enable streaming responses.")

    email_addr: str|None = Field(None, description="The email address for the email inbox.")

# class EmailBotConfig(BaseModel):
#     ''' Configuration schema for EmailBot
#     '''
#     email_addr: str = Field(..., description="The email address for the email inbox.")
#     llm_model: str|None = Field("deepseek/deepseek-chat", description="LLM model path or API.")




class EmailAgent():
    ''' The main EmailBot class
    '''
    def __init__(self, email_addr: str, llm_model: str|ChatLiteLLM):
        @tool(args_schema=EmailFormat)
        async def send_email(to_addr: str, subject: str, text: str, from_addr: str|None = None)-> ToolMessage:
            ''' Send an email'''
            if from_addr is None:
                from_addr = self.default_email_addr

            await simulate_streaming(f"Sending email from {from_addr} to {to_addr} with subject '{subject}'...")
            await simulate_streaming(f"Email content:\n{text}\n")
            ## Send Email
            self.client.inboxes.messages.send(
                inbox_id=from_addr,
                to=to_addr,
                subject=subject,
                text=text
            )
            return "Email Done"
        

        api_key = os.getenv("AGENTMAIL_API_KEY")

        # Create an inbox
        print("Creating inbox...")


        try:
            # Initialize the client
            self.client = AgentMail(api_key=api_key)

            if not email_addr:
                raise ValueError("Email address must be provided in the config.")
            
            self.inbox = self.client.inboxes.get(email_addr)

            self.default_email_addr = email_addr 

            # # self.inbox = self.client.inboxes.create() # domain is optional

            self.agent = create_agent(llm_model, tools=[send_email])

            self.send_email_tool = send_email

            # email_llm = (
            #     ChatLiteLLM(model=llm_model, streaming=False, temperature=0)
            #     if isinstance(llm_model, str)
            #     else llm_model
            # )

            # self.agent = email_llm.bind_tools([send_email])

            print("Inbox created successfully!")
            print(self.inbox)
            self.is_available=True
        except Exception as e:
            print("Email agent is not available! Error: ", e)
            self.is_available=False

    async def __call__(self, state: State):
        ''' Invoke the email agent with the latest message in the state '''
        print("Within Email Agent: ",state)
        print(type(state))
        result = await self.agent.ainvoke(state)
        print(result)
        
        return result#{"messages": [result]}
    
        # result = await self.agent.ainvoke(state["messages"][-1].content)
        # if isinstance(result, AnyMessage):
        #     last_message = result
        # else:
        #     last_message = result["messages"][-1]
        # return {"messages": [last_message]}


    




class ResearchBot():
    ''' The main ResearchBot class
    '''
    def __init__(self, config: ResearchBotConfig):
        @tool(return_direct=True)
        async def local_files_search(query):
            '''
                Searches information from the following sources:
                    1. local files
                    2. personal folders
                    3. knowledge base
                    4. personal knowledge base
            '''
            print(query)
            res = await self.collection.ainvoke(query)

            print("---------------- End of local_files_search ----------------")
            return res
        
        @tool(args_schema=ArxivSearch, return_direct=True)
        async def arxiv_search(keywords: Optional[List[str]] = None,
                        must_have_all_keywords: bool = True,
                        categories: Optional[List[str]] = None,
                        must_have_all_categories: bool = True,
                        authors: Optional[List[str]] = None,
                        must_have_all_authors: bool = True,
                        date_from: Optional[date] = None,
                        date_to: Optional[date] = None,
                        max_results: Optional[int] = 10,
                        sort_by: Optional[Literal["relevance", "lastUpdatedDate", "submittedDate"]] = "submittedDate",
                        markdown_format: bool = False
                        ) -> str:
            '''
                Searches the arxiv for relevant papers based on the user's query.
            '''
            return await self._arxiv_search(
                keywords, must_have_all_keywords, categories, must_have_all_categories,
                authors, must_have_all_authors, date_from, date_to, max_results, sort_by, markdown_format
            )

        self.config = config
        streaming = config.streaming
        is_local_embedding = config.is_local_embedding
        is_local_cross_encoder = config.is_local_cross_encoder

        if is_local_embedding:
            # Download into the project's models folder to avoid external network dependency
            local_model_path = f"./models/{config.embedding_model}" 

            if not Path(local_model_path).exists():
                print("Downloading embedding model to local path...")
                snapshot_download(
                    repo_id=config.embedding_model,
                    local_dir=local_model_path,
                    local_dir_use_symlinks=False
                )
            else:
                print("Using embedding model from local path:{}".format(config.embedding_model))
            embedding = local_model_path
        else:
            
            embedding = config.embedding_model

        if is_local_cross_encoder:
            local_cross_encoder_path = f"./models/{config.cross_encoder_model}"
            if not Path(local_cross_encoder_path).exists():
                print("Downloading cross encoder model to local path...")
                snapshot_download(
                    repo_id=config.cross_encoder_model,
                    local_dir=local_cross_encoder_path,
                    local_dir_use_symlinks=False
                )
            else:
                print("Using cross encoder model from local path:{}".format(config.cross_encoder_model))
            cross_encoder = local_cross_encoder_path
        else:
            cross_encoder = config.cross_encoder_model
        
        if config.llm_api_key is not None:
            config.llm_api_key = config.llm_api_key.strip()

            os.environ['OPENAI_API_KEY'] = config.llm_api_key
            os.environ['DEEPSEEK_API_KEY'] = config.llm_api_key
            os.environ['BAAI_API_KEY'] = config.llm_api_key
            os.environ['ANTHROPIC_API_KEY'] = config.llm_api_key
            os.environ['GEMINI_API_KEY'] = config.llm_api_key
            os.environ['QWEN_API_KEY'] = config.llm_api_key

        print(config)
        try:
            ## Initialize with your desired model and enable streaming
            self.main_llm = ChatLiteLLM(
                model=config.llm_model, # Or any LiteLLM-supported model
                streaming=streaming,
                temperature=0
            )

            self.assistant_llm= ChatLiteLLM(
                model=config.llm_model, # Or any LiteLLM-supported model
                streaming=False,
                temperature=0
            )
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            raise e

        ## TODO: make sure that the llm supports streaming and tool calls

        self.cross_encoder = cross_encoder
        self.embedding = embedding


        self.embedding, self.cross_encoder = get_embedding_and_cross_encoder(self.embedding, self.cross_encoder, device=get_device(), is_local_embedding=is_local_embedding, is_local_cross_encoder=is_local_cross_encoder)

        

        self.collection = Collection(llm=self.main_llm, embedding=self.embedding, reranker=self.cross_encoder, chunk_chars=self.config.chunk_chars, overlap=self.config.overlap, max_sources=self.config.max_sources, persist_directory=self.config.persist_directory, streaming=self.config.streaming)

        research_tools = [parse_fuzzy_date, arxiv_search, local_files_search, today]

        system_prompt = SystemMessage(
            content=config.system_prompt
        )

        # print("@Here! System prompt: ", system_prompt.content)


        self.arxiv_client = arxiv.Client(delay_seconds=4)

        ## by default this function actually create a ReAct agent graph
        self.research_agent = create_agent(
            self.main_llm,
            tools=research_tools,
            system_prompt=system_prompt
        )

        self.email_agent = EmailAgent(email_addr=config.email_addr, llm_model=self.main_llm)
        self.is_email_agent_availble = self.email_agent.is_available


        builder = StateGraph(State)

        def router_node(state: State) -> State:
            print("Starting to route...")
            return state
        
        # async def do_research_event(state: State):
        #     print("Starting to do research...")
        #     input_text = state.messages[-1].content
        #     await simulate_streaming("\n")
        #     res = await self.research_agent.ainvoke({"messages": content_expand_system_prompt.format(content=input_text)})
        #     return {"messages": [AIMessage(content=res)]}
        
        async def send_email_event(state: State):
            print("Starting to send email...")
            input_text = state.messages[-1].content
            # input_text = state["messages"][-1].content
            details = create_schema_by_tool_binding(self.assistant_llm, SendEmailDetails, email_information_extraction_system_prompt+input_text)
            to_addr = details.to_addr
            content_desc = details.content_description
            requires_search = details.requires_search

            print("Extract content description:" + content_desc)
            print("to_addr:", to_addr)
            print("Requires search:", requires_search)

            

            if requires_search:
                email_content = await self.research_agent.ainvoke({"messages": content_expand_system_prompt.format(content=content_desc)})
                email_content = get_content_from_response(email_content)
            else:
                email_content = content_desc


            print("Generated email content:" + email_content)
            res = await self.email_agent.send_email_tool.ainvoke({"to_addr": to_addr, "subject": "Email from ResearchBot", "text": email_content})

            return {"messages": [AIMessage(content=str(res))]}

        def router_conditions(state: State) -> Literal["do_research_event","send_email_event"]:
            input_text = state.messages[-1].content
            
            print("in router_conditions", type(state))
            ## First: to classify with rule-based patterns
            input_text_lower = input_text.strip().lower()

            
            

            ## Second: to classify with cross-encoder

            intent = ["searching arxiv papers", "search local files", "doing other tasks"]
            if self.is_email_agent_availble:
                intent.append("send an email to someone")

                strong_email_signals = [
                    r"^send .*email", 
                    r"^compose .*mail", 
                    r"send .* via email"
                ]
                for pattern in strong_email_signals:
                    if re.search(pattern, input_text_lower):
                        print(f"Router (Rule-based): Hit pattern '{pattern}' -> send_email_event")
                        return "send_email_event"

            scores = self.cross_encoder.predict([[input_text_lower, query] for query in  intent])
            best_intent_idx = int(np.argmax(scores))
            print(f"Router (Cross-Encoder): intent scores: {scores}, best intent idx: {intent[best_intent_idx]}")


            research_agent_indices = [i for i in range(len(intent))]
            send_email_event_indices = [len(intent)-1] if self.is_email_agent_availble else []
            
            if best_intent_idx in send_email_event_indices:
                return "send_email_event"
            elif best_intent_idx in research_agent_indices:
                return "do_research_event"
            else:
                return "do_research_event"
            

            
            # ## Third: to classify with structured output parsing with llm
            # try:
            #     choice = create_schema_by_tool_binding(self.main_llm, RouterChoice, input_text_lower)
            #     goto = choice.classification
            #     print(f"Router (Schema): reasoning: {choice.reasoning}, goto: {choice.classification}")

            #     # second way to output structured data
            #     # router_prompt = (
            #     #     '''You are the routing decision maker.
            #     #         Current members: do_research_event, send_email_event
            #     #         Based on the current context, decide the next step:

            #     #         Needs understanding requirements, planning, writing content → do_research_event
            #     #         Needs to write email, format email, send email → send_email_event
                        
            #     #         Return ONLY valid JSON like: {"reasoning":"...","classification":"do_research_event"}.
            #     #         classification must be exactly one of: "do_research_event" or "send_email_event".
            #     #     '''
            #     # )
            #     # resp = create_schema_output(self.main_llm, Router, input_text+router_prompt)
            #     # goto = json.loads(resp).get("classification")

            #     if goto not in ["do_research_event", "send_email_event"]:
            #         print(f"Router error: invalid goto '{goto}'")
            #         return "do_research_event"
            #     else:
            #         return goto
            # except Exception as e:
            #     print(f"Router error: {e}")
            #     return "do_research_event"


        builder.add_node("router_node", router_node)
        builder.add_node("do_research_event", self.research_agent)
        builder.add_node("send_email_event", send_email_event)
        builder.add_edge(START, "router_node")
        builder.add_conditional_edges("router_node", router_conditions)
        
        builder.add_edge("do_research_event", END)
        builder.add_edge("send_email_event", END)
        self.graph = builder.compile()

        # self.graph.astream_events

        display(Image(self.graph.get_graph().draw_mermaid_png()))



    


    async def aadd(self, local_file: str):
        ''' Create a new collection '''
        await self.collection.aadd(local_file)


    async def stream_response(self, input, callback=None, **kwargs):
        ''' Stream response from the ResearchBot agent '''
        # 使用 astream_events 来精细控制输出
        

        if isinstance(input, str):
            inputs = {"messages": [HumanMessage(content=input)]}
        else:
            inputs = input

        last_kind = None # 用于跟踪上一个事件类型

        async for event in self.graph.astream_events(inputs, version="v1"):
            kind = event["event"]
            
            # 只关心 chat_model 生成的块流 (on_chat_model_stream)
            if kind == "on_chat_model_stream":
                # [新增] 如果是从其他事件切换回来的（比如刚执行完工具），先加个换行美化一下
                if last_kind and last_kind != "on_chat_model_stream":
                     if callback: callback("\n")
                     print("\n", end="", flush=True)

                content = event["data"]["chunk"].content
                # 确保 content 不为空
                if content: 
                    if callback:
                        callback(content)
                    print(content, end="", flush=True)
                
                last_kind = kind # 更新状态

            # [新增] 监听工具结束事件，给工具输出前后加换行
            elif kind == "on_tool_end":
                if callback: callback("\n")
                print("\n", end="", flush=True)
                last_kind = kind

    # async def stream_response(self, input, callback=None, **kwargs):
    #     ''' Stream response from the ResearchBot agent '''
    #     # 使用 astream_events 来精细控制输出

    #     if isinstance(input, str):
    #         inputs = {"messages": [HumanMessage(content=input)]}
    #     else:
    #         inputs = input

    #     async for event in self.graph.astream_events(inputs, version="v1"):
    #         kind = event["event"]
            
    #         # 只关心 chat_model 生成的块流 (on_chat_model_stream)
    #         if kind == "on_chat_model_stream":
    #             content = event["data"]["chunk"].content
    #             # 确保 content 不为空，且不是工具调用请求
    #             if content: 
    #                 if callback:
    #                     callback(content)
    #                 print(content, end="", flush=True)

        ## simple stream
        # await stream_response(self.graph, input, streaming=self.config.streaming, **kwargs)

    async def ainvoke(self, input: str| list[HumanMessage]| State, **kwargs):
        ''' Asynchronously query the ResearchBot agent to get an answer '''
        if isinstance(input, State):
            return await self.graph.ainvoke(input, **kwargs)
        elif isinstance(input, list):
            return await self.graph.ainvoke(State(messages=input), **kwargs)
        else:
            return await self.graph.ainvoke(State(messages=[HumanMessage(content=input)]), **kwargs)
        # return await self.research_agent.ainvoke(input=input, **kwargs)
    
    def invoke(self, input: str| list[HumanMessage], **kwargs):
        ''' Synchronously query the ResearchBot agent to get an answer '''
        # return self.research_agent.invoke(input=input, **kwargs)
        if isinstance(input, State):
            return self.graph.invoke(input, **kwargs)
        elif isinstance(input, list):
            return self.graph.invoke(State(messages=input), **kwargs)
        else:
            return self.graph.invoke(State(messages=[HumanMessage(content=input)]), **kwargs)
        
    def set_callback(self, callback):
        ''' Set the callback function for streaming responses '''
        self.collection.set_callback(callback)
        self.callback = callback

    async def _process_single_arxiv(self, llm, title: str, full_summary: str, semaphore: asyncio.Semaphore) -> ArxivRefine|None:
        ''' Process a single document and add it to the vector store '''
        async with semaphore:
            try:
                res = create_schema_by_tool_binding(llm, ArxivRefine, refine_system_prompt.format(title=title, full_summary=full_summary))
                return res
                # ArxivRefine
            except Exception as e:
                print(f"Error processing arxiv paper with {title}: {e}")
                return None

    
    async def _arxiv_search(
                    self,
                    keywords: Optional[List[str]] = None,
                    must_have_all_keywords: bool = True,
                    categories: Optional[List[str]] = None,
                    must_have_all_categories: bool = True,
                    authors: Optional[List[str]] = None,
                    must_have_all_authors: bool = True,
                    date_from: Optional[date] = None,
                    date_to: Optional[date] = None,
                    max_results: Optional[int] = 10,
                    sort_by: Optional[Literal["relevance", "lastUpdatedDate", "submittedDate"]] = "submittedDate",
                    markdown_format: bool = False
                    ) -> str:
        '''
            Searches the arxiv for relevant papers based on the user's query.
        '''
        

        print("------------- arxiv_search ---------------")
        print("keywords:", keywords,
              "\ncategories:", categories,
              "\nauthors:", authors,
              "\ndate_from:", date_from,
              "\ndate_to:", date_to,
              "\nmax_results:", max_results,
              "\nsort_by:", sort_by
              )

        ## organize query parts
        query_parts = []


        if authors:

            authors_query_parts = []
            authors = [author.strip() for author in authors]
            diff_format_authors = [
                authors,
                [author.replace(" ", ", ",1).strip() for author in authors]
            ]
            for diff_format in diff_format_authors:
                if must_have_all_authors:
                    authors_query_parts.append(" AND ".join([f"au:\"{author}\"" for author in diff_format]))
                else:
                    authors_query_parts.append(" OR ".join([f"au:\"{author}\"" for author in diff_format]))
            query_parts.append(" OR ".join(["("+lim+")" for lim in authors_query_parts]))

            
            # query_parts.append(" OR ".join([f"au:\"{author}\"" for author in authors]))

            # if must_have_all_authors:
            #     query_parts.append(" AND ".join([f"au:\"{author}\"" for author in authors]))
            # else:
            #     query_parts.append(" OR ".join([f"au:\"{author}\"" for author in authors]))

            # query_parts.append(" OR ".join([f"au:\"{author}\"" for author in authors]))
            # query_parts.append(" OR ".join([f"(au:\"{author}\")" for author in authors]))
        if keywords:
            if must_have_all_keywords:
                query_parts.append(" AND ".join([f"all:\"{kw}\"" for kw in keywords]))
            else:
                query_parts.append(" OR ".join([f"all:\"{kw}\"" for kw in keywords]))
        
        
        
        
        
        if categories:
            if must_have_all_categories:
                query_parts.append(" AND ".join([f"cat:\"{cat}\"" for cat in categories]))
            else:
                query_parts.append(" OR ".join([f"cat:\"{cat}\"" for cat in categories]))
        
        if not date_from:
            date_from = date.fromordinal(date.today().toordinal()-14)
        
        if not date_to:
            date_to = date.fromordinal(date.today().toordinal())
        

        query_parts.append(f"submittedDate:[{convert_date_to_arxiv_format(date_from)} TO {convert_date_to_arxiv_format(date_to)}]")


        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate
        }

        query = "("+query_parts[0]+")" + " AND "
        query += " AND ".join(["("+query_parts[i]+")" for i in range(1,len(query_parts))])
        print(query)

        ## start arxiv searching
        search = arxiv.Search(
            query = query,
            max_results = max_results,
            sort_by = sort_map[sort_by]
        )

        tmp_results = []
        
        

        print(urlencode(search._url_args()))

        time_format = r"%Y-%m-%d %H:%M:%S UTC"

        for i, r in enumerate(self.arxiv_client.results(search)):
            tmp_results.append(ArxivResult(
                title=r.title,
                authors=[str(author) for author in r.authors],
                full_summary=r.summary,
                pdf_url=r.pdf_url,
                updated=r.updated.strftime(time_format),
                published=r.published.strftime(time_format),
                doi=r.doi,
                refined_summary="",
                score=0))
            
        arxiv_results = ArxivResults(
            query=query,
            total_results=len(tmp_results),
            papers=tmp_results
        )

        formatted_output = [f"Found {len(arxiv_results.papers)} papers for query: '{query}'\n"]
        
        arxiv_papers = list(arxiv_results.papers)
        
        

        if len(arxiv_papers) == 0:
            await simulate_streaming(text=f"\nNo papers found for query: {query}\n\n", callback=self.callback)
            return f"No papers found for query: {query}\n\n"
        else:
            await simulate_streaming(text=f"Found {len(arxiv_papers)} papers for query: {query}\n\n", callback=self.callback)

        
        
        # ## create refine chain
        # refine_chain = ChatPromptTemplate.from_template(abstract_refine_system_prompt) | self.assistant_llm

        # ## asynchronously refine abstracts
        # tasks = [summarize_abstract(refine_chain, paper.title, paper.full_summary) for paper in arxiv_papers]
        # refine_abstracts = await asyncio.gather(*tasks)
        # print(f"After refining {len(arxiv_papers)} papers concurrently...")


        ## TODO: add semaphore to limit the number of concurrent tasks to avoid OOM or rate limits.
        ## TODO: summarize abstract and extract keywords.
        ## create a chain to refine abstract and extract keywords
        batch_size = 4
        semaphore = asyncio.Semaphore(batch_size)  # Limit to 4 concurrent tasks

        for i in range(0, len(arxiv_papers), batch_size):
            batch_papers = arxiv_papers[i:i+batch_size]

            tasks = [self._process_single_arxiv(self.assistant_llm, paper.title, paper.full_summary, semaphore) for paper in batch_papers]
            refined_info = await tqdm_asyncio.gather(*tasks)

            await simulate_streaming(text="\n", callback=self.callback)

            ## Process papers (invoking LLM for summary extraction)
            for j, (paper, refined) in enumerate(zip(batch_papers, refined_info)):
                keywords_str = ", ".join(refined.keywords) if refined and refined.keywords else "N/A"
                refined_summary_str = refined.refined_summary if refined and refined.refined_summary else "N/A"
                paper_idx = i+j+1

                print(f"------------- Paper {paper_idx} ---------------")

                
                if markdown_format:
                    entry = (
                        f"### {paper_idx}. {paper.title}\n"
                        f"{', '.join(paper.authors)}\n"
                        f"- **Refined Summary (AI)**: {refined_summary_str}\n"
                        f"- **Keywords (AI): {keywords_str}\n"
                        f"- **Link**: {paper.pdf_url}\n"
                        f"- **Published**: {paper.published}  **Updated**: {paper.updated}\n"
                    )
                else:
                    if i==0:
                        s = f"\n"
                    else:
                        s = ""
                    s += (
                        f"Keywords (AI): \n{keywords_str}\n\n"
                        f"{paper_idx}. {paper.title}\n\n"
                        f"Authors: \n{', '.join(paper.authors)}\n\n"
                        f"Refined Summary (AI): \n{refined_summary_str}\n\n"
                        f"Link: {paper.pdf_url}\n\n"
                        f"Published: {paper.published}\n"
                        f"Updated: {paper.updated}\n"
                        
                        f"------------------------------------\n"
                    )
                    
                    entry = s
                    
                # print(entry)
                await simulate_streaming(text=entry, callback=self.callback)
                print(self.config.streaming, self.callback)
                
                formatted_output.append(entry)

        # await simulate_streaming(text="\n\n", callback=self.callback)

        print("------------- arxiv_search ---------------")

        return ("\n".join(formatted_output))+"\n\n"


    


async def main():
    pass
    # ''' for testing the Collection class '''
    # tools = []
    # streaming = True
    # is_local_embedding = True
    # is_local_cross_encoder = True

    # # Initialize with your desired model and enable streaming
    # main_llm = ChatLiteLLM(
    #     model="deepseek/deepseek-chat", # Or any LiteLLM-supported model
    #     streaming=streaming,
    #     temperature=0
    # )

    # assistant_llm= ChatLiteLLM(
    #     model="deepseek/deepseek-chat", # Or any LiteLLM-supported model
    #     streaming=False,
    #     temperature=0
    # )

    # # agent = create_agent(
    # #     main_llm,
    # #     tools = [parse_fuzzy_date, arxiv_search, local_files_search]
    # # )






    # if is_local_embedding:
    #     # Download into the project's models folder to avoid external network dependency
    #     local_model_path = "./models/BAAI/bge-m3" 
    #     # local_model_path = "./models/BAAI/bge-large-en" 

    #     if not Path(local_model_path).exists():
    #         snapshot_download(
    #             # repo_id="BAAI/bge-m3",
    #             # repo_id="BAAI/bge-base-en-v1.5",
    #             repo_id="BAAI/bge-small-en-v1.5",
    #             local_dir=local_model_path,
    #             local_dir_use_symlinks=False
    #         )
    #     embedding = local_model_path

    # if is_local_cross_encoder:
    #     local_cross_encoder_path = "./models/cross-encoder/ms-marco-MiniLM-L-6-v2"
    #     if not Path(local_cross_encoder_path).exists():
    #         snapshot_download(
    #             repo_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
    #             local_dir=local_cross_encoder_path,
    #             local_dir_use_symlinks=False
    #         )
    #     reranker = local_cross_encoder_path
    # collection = Collection(llm=main_llm, embedding=embedding, reranker=reranker, chunk_chars=500, overlap=100, max_sources=10)
    # await collection.aadd_dir("./my_papers")


if __name__ == "__main__":
    asyncio.run(main())



