import random

from app.config import PATH_METADATA
from datasets import load_dataset
import json


def read_metadata(path=PATH_METADATA):
    abstract_titles, abstract_texts = [], []

    with open(path, 'rb') as file:
        metadata = json.load(file)
        for data in metadata:
            abstract_titles.append(data['abstract_title'])
            abstract_texts.append(data['abstract_text'])

    return abstract_titles, abstract_texts


def prompt (question, context):
    prompt_text = f"""
            You are a model integrated into a retrieval-augmented generation (RAG) system
            designed to answer questions based on external documents.
            Here is the context retrieved from the documents:
            {context}
            Using this context, provide a concise and accurate answer to the question below.
            If the context does not contain enough information, respond with
            "I don't have enough information to answer this question."
            Do not invent or include any information not found in the documents.
            If the necessary information is found in the provided documents, use it to construct your
            response and include a citation by referencing the document number in round brackets.
            For example:  "Natural Language Processing (NLP) is a field of artificial intelligence
            that enables machines to understand, interpret, and generate human language.(3)."
            Question: {question}
    """
    return prompt_text


def random_article():
    ds = load_dataset("pt-sk/research_papers_short")
    my_docs = ds['train'].select(range(1000))
    random_index = random.randint(0, len(my_docs) - 1)
    return my_docs[random_index]['title'], my_docs[random_index]['abstract']