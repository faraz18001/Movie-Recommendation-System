import pandas as pd
import numpy as np
import ast
import os
from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS as LangChainFAISS
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os
import sys
import subprocess
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
# Set up environment variables
os.environ['OPENAI_API_KEY'] = ''
api_key = os.getenv('OPENAI_API_KEY')

# File paths
DATA_FILE = 'movies_metadata.csv'
FAISS_INDEX_FILE = 'faiss_movie_index.pkl'

# Data importing and preprocessing
def load_and_preprocess_data():
    md = pd.read_csv(DATA_FILE, low_memory=False)
    md['genres'] = md['genres'].apply(ast.literal_eval)
    md['genres'] = md['genres'].apply(lambda x: [genre['name'] for genre in x])
    
    def calculate_weighted_rate(vote_average, vote_count, min_vote_count=10):
        return (vote_count / (vote_count + min_vote_count)) * vote_average + (min_vote_count / (vote_count + min_vote_count)) * 5.0

    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    min_vote_count = vote_counts.quantile(0.95)

    md['weighted_rate'] = md.apply(lambda row: calculate_weighted_rate(row['vote_average'], row['vote_count'], min_vote_count), axis=1)
    md.dropna(inplace=True)
    md_final = md[['genres', 'title', 'overview', 'weighted_rate']].reset_index(drop=True)
    md_final['text'] = md_final.apply(lambda row: f"Title: {row['title']}. Overview: {row['overview']} Genres: {', '.join(row['genres'])}. Rating: {row['weighted_rate']}", axis=1)
    return md_final

# Load or create FAISS index with LangChain
def load_or_create_faiss_index(data):
    if os.path.exists(FAISS_INDEX_FILE):
        print("Loading existing FAISS index...")
        return LangChainFAISS.load_local(FAISS_INDEX_FILE, OpenAIEmbeddings())
    else:
        print("Creating new FAISS index...")
        documents = [Document(page_content=text, metadata={"title": title, "genres": genres, "rating": rating}) 
                     for text, title, genres, rating in zip(data['text'], data['title'], data['genres'], data['weighted_rate'])]
        vectorstore = LangChainFAISS.from_documents(documents, OpenAIEmbeddings())
        vectorstore.save_local(FAISS_INDEX_FILE)
        return vectorstore

# Main execution
print("Loading and preprocessing data...")
md_final = load_and_preprocess_data()

print("Setting up FAISS index...")
docsearch = LangChainFAISS.load_local(FAISS_INDEX_FILE, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
# Set up RetrievalQA chain
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=docsearch.as_retriever())

# Example usage
query = "I'm looking for an animated action movie. What could you suggest to me?"
result = qa_chain({"query": query})
print(f"\nQuery: {query}")
print(f"Answer: {result['result']}")

# Similarity search example
print("\nTop 3 similar movies:")
docs = docsearch.similarity_search(query, k=3)
for doc in docs:
    print(f"Title: {doc.metadata['title']}")
    print(f"Genres: {doc.metadata['genres']}")
    print(f"Rating: {doc.metadata['rating']}")
    print(f"Content: {doc.page_content[:100]}...")  # Truncated for brevity
    