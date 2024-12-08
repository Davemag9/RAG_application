import json, faiss
import numpy as np
from rank_bm25 import BM25Okapi
from app.config import PATH_BASE, PATH_PREPROCESSED_DOCS
from app.scripts.dataset_to_base import text_preprocessing

class Search:
    def __init__(self, path_base=PATH_BASE, path_preprocessed=PATH_PREPROCESSED_DOCS):
        self.bm25_model = BM25Okapi([doc.split(" ") for doc in self.read_preprocessed_docs(path_preprocessed)])
        self.vector_search = faiss.read_index(path_base)


    def read_preprocessed_docs(self, path_preprocessed):
        with open(path_preprocessed, 'rb') as f:
            docs = json.load(f)
        return docs


    def search_bm25(self, query, k=3):

        text = text_preprocessing(query).split(" ")
        scores = self.bm25_model.get_scores(text)
        sorted_scores = np.argsort(scores)[::-1]

        return sorted_scores[:k].tolist()


    def search_vector(self, query, k=3):

        tmp, indexes = self.vector_search.search(query, k)

        return indexes
