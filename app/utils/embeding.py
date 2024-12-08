from sentence_transformers import SentenceTransformer

class Embedding:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, revision="main")

    def query_embedding(self, query):
        query_embed = self.model.encode([query], normalize_embeddings=True)
        return query_embed

    def get_embedding(self, texts):
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings