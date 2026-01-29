from litellm import embedding
from prompt import *
from utility import *
from tools import *

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
        default="[⁰, ¹, ², ³, ⁴, ⁵, ⁶, ⁷, ⁸, ⁹]",
        description="An example citation key to illustrate the format."
    )


class ArxivSearch(BaseModel):
  ''' Schema for searching arxiv papers.
  '''
  keywords: Optional[List[str]] | None = Field(None, description="the main search keywords.")
  must_have_all_keywords: bool = Field(True, description="Whether all keywords must be present in the search results.")
  categories: Optional[List[str]] | None = Field(None, description="List of arxiv categories to search in")
  must_have_all_categories: bool = Field(True, description="Whether all categories must be present in the search results.")
  authors: Optional[List[str]] | None = Field(None, description="List of authors to search for")
  must_have_all_authors: bool = Field(True, description="Whether all authors must be present in the search results.")
  date_from: Optional[date] | None = Field(None, description="Start date in 'YYYY-MM-DD' format. Example: '2023-01-01'.")
  date_to: Optional[date] | None = Field(None, description="End date in 'YYYY-MM-DD' format. Defaults to today.")
  max_results: Optional[int] | None = Field(100, description="Maximum number of results to return")
  sort_by: Optional[Literal["relevance", "lastUpdatedDate", "submittedDate"]] | None = Field("submittedDate", description="Criterion to sort results by")

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
            target_dir = Path(dir_path)
        else:
            target_dir = dir_path

        all_files = [p for p in target_dir.rglob('*') if p.is_file() and self.is_solvable_file(p)]
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
        

        # Define a callback function to handle streaming output
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
            doc.metadata["relevance_score"] = float(score) # save the score
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

        # print(cite_ids)


        citation_str = "\n".join([citations[cite_ids[i]] for i in range(len(cite_ids))])+"\n\n"

        # print(main_content)
        # print(citation_str)


        return main_content + citation_str







    

class ResearchBotConfig(BaseModel):
    ''' Configuration schema for ResearchBot
    '''
    is_local_llm: bool = Field(False, description="Whether to use local LLM model.")
    is_local_embedding: bool = Field(True, description="Whether to use local embedding model.")
    is_local_reranker: bool = Field(True, description="Whether to use local reranker model.")
    llm_model: str = Field("deepseek/deepseek-chat", description="LLM model path or API.")
    embedding_model: str = Field("BAAI/bge-m3", description="Embedding model path or API.")
    reranker_model: str = Field("cross-encoder/ms-marco-MiniLM-L-6-v2", description="Reranker model path or API.")


    max_sources: int = Field(10, description="Maximum number of sources to retrieve.")
    chunk_chars: int = Field(500, description="Number of characters per document chunk.")
    overlap: int = Field(100, description="Number of overlapping characters between chunks.")
    persist_directory: str = Field("./output/collection", description="Directory to persist the collection data.")
    local_files_dir: str = Field(".", description="Directory path for local files to search.")
    is_streaming: bool = Field(True, description="Whether to enable streaming responses.")
    



class ResearchBot():
    ''' The main ResearchBot class
    '''
    def __init__(self, config: ResearchBotConfig):
        @tool
        async def local_files_search(query) -> str:
            '''
                Searches information in the local files or personal folders based on the user's query.
            '''
            print(query)
            res = await self.collection.ainvoke(query)

            print("---------------- End of local_files_search ----------------")
            return res
        
        @tool(args_schema=ArxivSearch)
        async def arxiv_search(keywords: Optional[List[str]] = None,
                        must_have_all_keywords: bool = True,
                        categories: Optional[List[str]] = None,
                        must_have_all_categories: bool = True,
                        authors: Optional[List[str]] = None,
                        must_have_all_authors: bool = True,
                        date_from: Optional[date] = None,
                        date_to: Optional[date] = None,
                        max_results: Optional[int] = 10,
                        sort_by: Optional[Literal["relevance", "lastUpdatedDate", "submittedDate"]] = "submittedDate"
                        ) -> str:
            '''
                Searches the arxiv for relevant papers based on the user's query.
            '''
            return await self._arxiv_search(
                keywords, must_have_all_keywords, categories, must_have_all_categories,
                authors, must_have_all_authors, date_from, date_to, max_results, sort_by
            )

        self.config = config
        is_streaming = config.is_streaming
        is_local_embedding = config.is_local_embedding
        is_local_reranker = config.is_local_reranker

        if is_local_embedding:
            # Download into the project's models folder to avoid external network dependency
            local_model_path = f"./models/{config.embedding_model}" 

            if not Path(local_model_path).exists():
                snapshot_download(
                    repo_id=config.embedding_model,
                    local_dir=local_model_path,
                    local_dir_use_symlinks=False
                )
            embedding = local_model_path

        if is_local_reranker:
            local_cross_encoder_path = f"./models/{config.reranker_model}"
            if not Path(local_cross_encoder_path).exists():
                snapshot_download(
                    repo_id=config.reranker_model,
                    local_dir=local_cross_encoder_path,
                    local_dir_use_symlinks=False
                )
            reranker = local_cross_encoder_path
        

        # Initialize with your desired model and enable streaming
        self.main_llm = ChatLiteLLM(
            model=config.llm_model, # Or any LiteLLM-supported model
            streaming=is_streaming,
            temperature=0
        )

        self.academic_llm= ChatLiteLLM(
            model=config.llm_model, # Or any LiteLLM-supported model
            streaming=False,
            temperature=0
        )

        self.collection = Collection(llm=self.main_llm, embedding=embedding, reranker=reranker, chunk_chars=self.config.chunk_chars, overlap=self.config.overlap, max_sources=self.config.max_sources, persist_directory=self.config.persist_directory)

        self.agent = create_agent(
            self.main_llm,
            tools = [parse_fuzzy_date, arxiv_search, local_files_search]
        )

    async def aadd_dir(self, local_files_dir: str):
        ''' Create a new collection '''
        await self.collection.aadd_dir(local_files_dir)

    async def stream_response(self, input: str, **kwargs):
        ''' Stream response from the ResearchBot agent '''
        await stream_response(self.agent, input, streaming=self.config.is_streaming, **kwargs)

    async def ainvoke(self, input: str, **kwargs):
        ''' Asynchronously query the ResearchBot agent to get an answer '''
        return await self.agent.ainvoke(input=input, **kwargs)
    
    def invoke(self, input: str, **kwargs):
        ''' Synchronously query the ResearchBot agent to get an answer '''
        return self.agent.invoke(input=input, **kwargs)

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
                    sort_by: Optional[Literal["relevance", "lastUpdatedDate", "submittedDate"]] = "submittedDate"
                    ) -> str:
        '''
            Searches the arxiv for relevant papers based on the user's query.
        '''
        

        print("------------- arxiv_search ---------------")

        ## organize query parts

        query_parts = []


        if authors:

            authors_query_parts = []
            authors = [author.strip() for author in authors]
            diff_format_authors = [
                authors,
                # [author.replace(",", " ").strip() if "," in author else ", ".join(author.split()).strip()
                # for author in authors],
                [author.replace(" ", ", ").strip() for author in authors]
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

        query = query_parts[0] + " AND "
        query += " AND ".join(["("+query_parts[i]+")" for i in range(1,len(query_parts))])
        # query = " AND ".join(["("+lim+")" for lim in query_parts])
        print(query)

        ## start arxiv searching

        search = arxiv.Search(
            query = query,
            max_results = max_results,
            sort_by = sort_map[sort_by]
        )

        tmp_results = []
        
        client = arxiv.Client()

        print(urlencode(search._url_args()))

        for i, r in enumerate(client.results(search)):
            tmp_results.append(ArxivResult(
                title=r.title,
                authors=[str(author) for author in r.authors],
                full_summary=r.summary,
                pdf_url=r.pdf_url,
                updated=r.updated.isoformat(),
                published=r.published.isoformat(),
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
        
        ## check if no papers found
        if not arxiv_papers:
            return f"No papers found for query: {query}\n\n"
        
        ## create refine chain
        refine_chain = ChatPromptTemplate.from_template(abstract_refine_system_prompt) | self.academic_llm

        ## asynchronously refine abstracts
        tasks = [summarize_abstract(refine_chain, paper.title, paper.full_summary) for paper in arxiv_papers]
        refine_abstracts = await asyncio.gather(*tasks)
        print(f"After refining {len(arxiv_papers)} papers concurrently...")


        ## Process papers (invoking LLM for summary extraction)
        for i, (paper, refined_summary) in enumerate(zip(arxiv_papers, refine_abstracts)):
            entry = (
                f"### {i+1}. {paper.title}\n"
                f"{', '.join(paper.authors)}\n"
                f"- **Refined Summary (AI)**: {refined_summary}\n"
                f"- **Link**: {paper.pdf_url}\n"
                f"- **Published**: {paper.published}  **Updated**: {paper.updated}\n"
            )
            print(entry)
            formatted_output.append(entry)

        print("------------- arxiv_search ---------------")

        return ("\n".join(formatted_output))+"\n\n"


    


async def main():
    ''' for testing the Collection class '''
    tools = []
    is_streaming = True
    is_local_embedding = True
    is_local_reranker = True

    # Initialize with your desired model and enable streaming
    main_llm = ChatLiteLLM(
        model="deepseek/deepseek-chat", # Or any LiteLLM-supported model
        streaming=is_streaming,
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
        # Download into the project's models folder to avoid external network dependency
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
    collection = Collection(llm=main_llm, embedding=embedding, reranker=reranker, chunk_chars=500, overlap=100, max_sources=10)
    await collection.aadd_dir("./my_papers")


if __name__ == "__main__":
    asyncio.run(main())



