from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
import pinecone
import gradio as gr
import os






def read_doc(directory: str):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents


def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    return chunks


def retrieve_query(index, query, k=2):
    matching_results = index.similarity_search(query, k=k)
    return matching_results


def retrieve_answers(index, chain, query):
    doc_search = retrieve_query(index, query)
    response = chain.run(input_documents=doc_search, question=query)
    return response


def qa_manager(query):
    return retrieve_answers(query)




