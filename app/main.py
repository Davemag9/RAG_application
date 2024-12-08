from app.utils.embeding import Embedding
from app.utils.helpful_functions import read_metadata, prompt, random_article
from app.utils.llm_config import LLM
from app.utils.search import Search


def run_RAG(query, search_type, API_KEY='3uTKE448T558Qmem6pBSbvW54nHBR4FP6Xnn6jCl'):

    llm = LLM(API_KEY)
    model = Embedding()
    embedding = model.query_embedding(query)

    search_model = Search()
    abstract_titles, abstract_texts = read_metadata()

    search_list = []

    match search_type:
        case 'bm25':
            search_list = search_model.search_bm25(query, 3)
        case 'vector_base':
            search_list = search_model.search_vector(embedding)[0].tolist()
        case 'None':
            return llm.generate_answer(f'Answer on question by 1-2 sentence, no punct: {query},'
                                       f' do not invent any information')


    context = ""
    for i, index in enumerate(search_list):
        context += " [" + str(i + 1) + "] " + abstract_texts[index]

    question = prompt(query, context)

    result = llm.generate_answer(question)
    if "I don't have enough information to answer this question." in result:
        return result

    return result, context



if __name__ == '__main__':
    # text = 'what evidence lower bound produce When used as a surrogate objective for maximum likelihood estimation in latent variable model'

    # print(run_RAG(text, 'vector_base')) #return answer and context files
    # print(run_RAG(text, 'bm25')) #return answer and context files
    # print(run_RAG(text, 'None')) #return answer, no context files
    # print(run_RAG("WHO iS JOE BIDEN", 'bm25')) #return answer, no context files
    # random_article() #return 1 random title with article

    # search_type only None, bm25 or vector_base