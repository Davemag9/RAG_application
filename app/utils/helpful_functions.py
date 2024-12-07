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

    return abstract_texts, abstract_titles