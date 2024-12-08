import json
from app.config import *
from datasets import load_dataset

import nltk
from nltk.tokenize import sent_tokenize

from app.utils.helpful_functions import get_metadata

nltk.download('punkt_tab')

from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem import PorterStemmer

import faiss
from app.utils.embeding import Embedding


def text_tokenization(text):
    
    text = text.replace(",", "")
    text = text.replace(";", "")
    text = text.replace(":", "")
    tokens = text.lower().split(' ')

    return tokens


def text_cleaner(text):

    tokens = text_tokenization(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in tokens if word not in stop_words]

    return filtered_words


def text_simplifier(tokens):

    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    text = " ".join(stemmed_tokens)
    
    return text


def text_preprocessing(text):
    
    filtered_words = text_cleaner(text)
    text = text_simplifier(filtered_words)

    return text


def write_preprocessed_docs(preprocessed_docs):
     
     with open(PATH_PREPROCESSED_DOCS, "w") as f:
        json.dump(preprocessed_docs, f, indent=4)


def docs_preprocessing(docs):

    preprocessed_docs = [text_preprocessing(doc) for doc in docs]
    write_preprocessed_docs(preprocessed_docs)


def get_chunks_and_metadata(docs, max_tokens):

    chunked_texts, metadata = [], []

    for _, text in enumerate(docs):
        sentences = sent_tokenize(text['abstract'])
        current_chunk = []
        current_chunk_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            if current_chunk_length + sentence_length > max_tokens:
                chunk = " ".join(current_chunk)
                chunked_texts.append(chunk)
                metadata.append({'abstract_title': text['title'], 'abstract_text': chunk})
                
                current_chunk = []
                current_chunk_length = 0

            current_chunk.append(sentence)
            current_chunk_length += sentence_length

        if current_chunk_length != 0:
            chunk = " ".join(current_chunk)
            chunked_texts.append(chunk)
            metadata.append({'abstract_title': text['title'], 'abstract_text': chunk})

    return chunked_texts, metadata


def write_metadata(metadata):
    with open(PATH_METADATA, "w") as f:
        json.dump(metadata, f, indent=4)


def create_base_metadata(docs, model: Embedding, dimension = 384):
    chunks, metadata = get_chunks_and_metadata(docs, 256)
    embeddings = model.get_embedding(chunks)

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, metadata


def write_base(base):
    faiss.write_index(base, PATH_BASE)


def script1():
    ds = load_dataset("pt-sk/research_papers_short")
    my_docs = ds['train'].select(range(100))

    print("complete1")
    model = Embedding()
    base, metadata = create_base_metadata(my_docs, model)

    print("complete2")
    write_metadata(metadata)

    print("complete3")
    write_base(base)


def script2():
    tmp, abstract_texts  = get_metadata()

    print("complete4")
    docs_preprocessing(abstract_texts)


def run_scripts():
    script1()
    print("Script1 Complete")
    script2()
    print("Script2 Complete")


if __name__ == '__main__':
    run_scripts()