from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import get_buffer_string
from operator import itemgetter
from multiquery import generate_queries_decomposition
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from prompts import question_prompt, qa_prompt
from configs import collection_name,threshold,temperature
from langchain_core.prompts import ChatPromptTemplate
from get_retriever import get_db
from langchain.load import dumps, loads
from langchain_openai import ChatOpenAI
from langchain.schema import Document
import json

class QueryProcessor:
    def __init__(self):
        self.sources = []
        self.data_urls = []
        self.top_chunks = []
        
    def clear_sources(self):
        self.sources = []
        
    def add_sources(self, source):
        self.sources.append(source)
        
    def clear_data_urls(self):
        self.data_urls = []
        
    def add_data_url(self, url):
        self.data_urls.append(url)
        
    def clear_top_chunks(self):
        self.top_chunks = []
        
    def add_top_chunks(self, chunks):
        self.top_chunks.append(chunks)

processor = QueryProcessor()

def _combine_documents(docs, document_separator="\n\n"):
    doc_strings = []
    processor.clear_sources()
    for doc in docs:
        # Remove curly braces from doc.page_content
        clean_content = doc.page_content.replace("{", "").replace("}", "")
        processor.add_sources(doc.metadata['source'])
        chunk = "content: \n" + clean_content + "\n\n"
        doc_strings.append(chunk)
    return document_separator.join(doc_strings)

def document_to_dict(doc):
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
    }

def dict_to_document(doc_dict):
    return Document(
        page_content=doc_dict["page_content"],
        metadata=doc_dict["metadata"],
    )
    
def get_unique_union(documents: list[list]):
    processor.clear_top_chunks()
    
    flattened_docs = []
    for sublist in documents:
        # Append page_content of top 2 chunks from each sublist to top_chunks
        processor.add_top_chunks([doc.page_content for doc in sublist[:2]])
        
        # Convert each Document to a dictionary before serializing
        flattened_docs.extend([dumps(document_to_dict(doc)) for doc in sublist])
    
    unique_docs = list(set(flattened_docs))
    return [dict_to_document(loads(doc)) for doc in unique_docs]

def create_content_jsons(docs, question, context, chat_history):
    if not isinstance(docs[0], Document):
        docs = get_unique_union(docs)
    text_image_jsons = []
    text_image_jsons.append({"type": "text", "text": qa_prompt.format(context=context, question=question, chat_history=chat_history)})

    processor.clear_data_urls()
    for doc in docs:
        images = json.loads(doc.metadata.get("data_urls", "[]"))
        for image in images:
            processor.add_data_url(image)
    
    print(f"Created content JSON with {len(text_image_jsons)} items")
    
    return json.dumps(text_image_jsons)

async def run_chain(memory, inputs):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    vectordb = get_db(collection_name=collection_name)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={'score_threshold': threshold})
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(question_prompt)

    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    }

    query_decompose = {
        "queries": itemgetter("standalone_question") | generate_queries_decomposition,
        "question": lambda x: x["standalone_question"],
    }

    retrieved_documents = {
        "docs": itemgetter("queries") | retriever.map() | get_unique_union,
        "question": lambda x: x["question"],
    }

    def prepare_final_inputs(x):
        return {
            "json_": create_content_jsons(x["docs"], x["question"], _combine_documents(x["docs"]), x["chat_history"]),
        }

    answer = {
        "answer": RunnablePassthrough() | prepare_final_inputs 
    }

    final_chain = loaded_memory | standalone_question | query_decompose | retrieved_documents | loaded_memory| answer
    
    result_chain = await final_chain.ainvoke(inputs)

    final_prompt = ChatPromptTemplate.from_messages([
        ("human", json.loads(result_chain["answer"]["json_"]))
    ])
    
    end_chain = final_prompt | ChatOpenAI(temperature=0, model="gpt-4o") | StrOutputParser()
    answer = await end_chain.ainvoke({})
    result = {
        "answer": answer,
        "sources": list(set(processor.sources)),  # Convert set to list
        "top2_chunks": processor.top_chunks,
        "images": list(set(processor.data_urls))
    }
    return answer, result
