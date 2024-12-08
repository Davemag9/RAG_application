from app.utils.embeding import Embedding
from app.utils.helpful_functions import get_metadata, prompt
from app.utils.llm_config import LLM
from app.utils.search import Search


def run_RAGul(query, search_type, API_KEY='3uTKE448T558Qmem6pBSbvW54nHBR4FP6Xnn6jCl'):

    llm = LLM(API_KEY)
    model = Embedding()
    embedding = model.query_embedding(query)

    search_model = Search()
    abstract_titles, abstract_texts = get_metadata()

    search_list = []

    match search_type:
        case 'bm25':
            search_list = search_model.search_bm25(query, 10)
        case 'vector_base':
            search_list = search_model.search_vector(query)[0].tolist()
        case _:
            raise ValueError(f"Invalid search type: {search_type}")

    context = ""
    for i, index in enumerate(search_list):
        context += " [" + str(i + 1) + "] " + abstract_texts[index]

    question = prompt(query, context)

    return llm.generate_answer(question), context



if __name__ == '__main__':
    text = 'what evidence lower bound produce When used as a surrogate objective for maximum likelihood estimation in latent variable models?'

    # "I don't have enough information to answer this question."
    print(run_RAGul(text, 'vector_base'))

