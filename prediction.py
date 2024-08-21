# For sample predict functions for popular libraries visit https://github.com/opendatahub-io/odh-prediction-samples

# Import libraries
# import tensorflow as tf

from langchain_community.llms import HuggingFaceHub
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
import pickle as pkl
import numpy as np
import streamlit as st
# Module from the Langchain library that provides embeddings for text processing using OpenAI language models.
# from langchain.embeddings.openai import OpenAIEmbeddings
# Python built-in module for handling temporary files.
import tempfile
# Python built-in module for time-related operations.
import time
# Below are the classes from the Langchain library
# from langchain import OpenAI, PromptTemplate, LLMChain
from langchain import PromptTemplate, LLMChain
# class from the Langchain library that splits text into smaller chunks based on specified parameters.
from langchain.text_splitter import CharacterTextSplitter
# This is a class from the Langchain library that loads PDF documents and splits them into pages.
from langchain.document_loaders import PyPDFLoader
# This is a function from the Langchain library that loads a summarization chain for generating summaries.
from langchain.chains.summarize import load_summarize_chain
# This is a class from the Langchain library that represents a document.
from langchain.docstore.document import Document
# This is a class from the Langchain library that provides vector indexing and similarity search using FAISS.
from langchain.vectorstores import FAISS
# This is a function from the Langchain library that loads a question-answering chain for generating answers to questions.
from langchain.chains.question_answering import load_qa_chain

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# Load your model.
# model_dir = 'models/myfancymodel'
# saved_model = tf.saved_model.load(model_dir)
# predictor = saved_model.signatures['default']


# Write a predict function 

def predict(args_dict):
#     arg = args_dict.get('arg1')
#     predictor(arg)
    print("LOGGING IN")
    login(token="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")
    query = args_dict.get('data')
    print("Flow : inside predict with query",query)
    
    # model = AutoModelForCausalLM.from_pretrained("mistralmnemo")
    # tokenizer = AutoTokenizer.from_pretrained("mistraltnemo")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407")
    # if(os.path.isdir("/mistral-7b-model")):
    #     print("model present")
    #     model = AutoModelForCausalLM.from_pretrained("/mistral-7b-model")
    # if(os.path.isdir("/mistral-7b-tokenizer")):
    #     tokenizer = AutoTokenizer.from_pretrained("/mistral-7b-tokenizer")
    # else:
    #     print("model NOT present")
    #     tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    #     model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    
    model.save_pretrained("models/mistralmnemo")
    tokenizer.save_pretrained("models/mistraltnemo")

    text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
    device=0
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    pages=[]
    for file in os.listdir("files/"):
        if file.endswith('.pdf'):
            pdf_path = os.path.join("files/", file)
            loader = PyPDFLoader(pdf_path)
            pages.extend(loader.load_and_split())
            

    # np_pages = np.array(pages)
    # np.save('pages',np_pages)
    # pages = np.load("pages.npy",allow_pickle=True)
    
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1400,
        chunk_overlap  = 100,
        length_function = len,
    )

    combined_content = ''.join([p.page_content for p in pages])
    texts = text_splitter.split_text(combined_content)
    document_search = FAISS.from_texts(texts, HuggingFaceEmbeddings(model_name="BAAI/bge-m3"))
    # document_search = FAISS.from_texts(texts, embed_model)
    docs = document_search.similarity_search(query)
    chain = load_qa_chain(llm, chain_type="stuff")
    # chain.run(input_documents=docs, question=query)
    return {'prediction': chain.run(input_documents=docs, question=query)}


def eci_predict(args_dict):
#     arg = args_dict.get('arg1')
#     predictor(arg)
    print("LOGGING IN")
    login(token="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")
    query = args_dict.get('text').replace('eciai', '')
    print("Flow : inside predict with query",query)
    
    # model = AutoModelForCausalLM.from_pretrained("mistralmnemo")
    # tokenizer = AutoTokenizer.from_pretrained("mistraltnemo")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407")
    # if(os.path.isdir("/mistral-7b-model")):
    #     print("model present")
    #     model = AutoModelForCausalLM.from_pretrained("/mistral-7b-model")
    # if(os.path.isdir("/mistral-7b-tokenizer")):
    #     tokenizer = AutoTokenizer.from_pretrained("/mistral-7b-tokenizer")
    # else:
    #     print("model NOT present")
    #     tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    #     model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    
    model.save_pretrained("models/mistralmnemo")
    tokenizer.save_pretrained("models/mistraltnemo")

    text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
    device=0
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    pages=[]
    for file in os.listdir("files/"):
        if file.endswith('.pdf'):
            pdf_path = os.path.join("files/", file)
            loader = PyPDFLoader(pdf_path)
            pages.extend(loader.load_and_split())
            

    # np_pages = np.array(pages)
    # np.save('pages',np_pages)
    # pages = np.load("pages.npy",allow_pickle=True)
    
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1400,
        chunk_overlap  = 100,
        length_function = len,
    )

    combined_content = ''.join([p.page_content for p in pages])
    texts = text_splitter.split_text(combined_content)
    document_search = FAISS.from_texts(texts, HuggingFaceEmbeddings(model_name="BAAI/bge-m3"))
    # document_search = FAISS.from_texts(texts, embed_model)
    docs = document_search.similarity_search(query)
    chain = load_qa_chain(llm, chain_type="stuff")
    # chain.run(input_documents=docs, question=query)
    return {'prediction': chain.run(input_documents=docs, question=query)}