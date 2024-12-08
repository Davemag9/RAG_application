from app.config import PATH_METADATA
from app.utils.embeding import Embedding
from app.utils.llm_config import LLM
import json

def get_model():
    return Embedding()


def get_llm(api_key):
    return LLM(api_key)


def get_metadata(path=PATH_METADATA):
    abstract_titles, abstract_texts = [], []

    with open(path, 'rb') as file:
        metadata = json.load(file)
        for data in metadata:
            abstract_titles.append(data['abstract_title'])
            abstract_texts.append(data['abstract_text'])

    return abstract_titles, abstract_texts


def prompt (question, context):
    system_prompt = f"""You are a language model integrated into a retrieval-augmented generation (RAG) system.
        Your task is to answer the user's query strictly based on the provided documents. Do not invent, speculate, or include any information not found in the documents.

        If the required information is available in the documents, use it to construct your response and cite the source by indicating the document number in square brackets. For example:
        DL stands for Deep Learning, a subset of Machine Learning that involves learning complex non-linear relationships between large datasets [6].

        If the information required to answer the query is not available in the documents, explicitly state:
        "The required information is not available in the provided documents."

        Ensure that:
        - The response is entirely based on the content of the documents.
        - Citations are accurate and directly linked to the information being cited.
        - No assumptions, speculations, or fabricated details are included.

        User query: {question}
        Documents:
        {context}
        """
    # prompt_text = f"""
    #         You are a model integrated into a retrieval-augmented generation (RAG) system
    #         designed to answer questions based on external documents.
    #         Here is the context retrieved from the documents:
    #         {context}
    #         Using this context, provide a concise and accurate answer to the question below.
    #         If the context does not contain enough information, respond with
    #         "I don't have enough information to answer this question."
    #         Do not invent or include any information not found in the documents.
    #         If the necessary information is found in the provided documents, use it to construct your
    #         response and include a citation by referencing the document number in square brackets.
    #         For example:  "Natural Language Processing (NLP) is a field of artificial intelligence
    #         that enables machines to understand, interpret, and generate human language.[3]."
    #         Question: {question}
    # """
    # return prompt_text
    return system_prompt