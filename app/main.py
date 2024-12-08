from app.utils.embeding import Embedding
from app.utils.helpful_functions import read_metadata, prompt, random_article
from app.utils.llm_config import LLM
from app.utils.search import Search
# import gradio as gr


def run_RAG(query, search_type, API_KEY='3uTKE448T558Qmem6pBSbvW54nHBR4FP6Xnn6jCl'): #Щоб не шукати ключ, залишили його вам

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


import gradio as gr

# Assuming run_RAG is defined above or imported
def run_RAG_gradio_interface(query, search_type):
    result = run_RAG(query, search_type)
    if isinstance(result, tuple):
        # If result and context are returned
        return result[0], result[1]
    else:
        # If only the result is returned
        return result, None


# Define the Gradio interface
def main():
    with gr.Blocks() as app:
        gr.Markdown("## RAG System Interface")

        with gr.Row():
            query = gr.Textbox(label="Enter your query", placeholder="Type your question here...")
            search_type = gr.Dropdown(
                choices=["None", "bm25", "vector_base"],
                label="Select Search Type",
                value="None"
            )

        with gr.Row():
            result_output = gr.Textbox(label="Answer", interactive=False)
            context_output = gr.Textbox(label="Context", interactive=False)

        submit_button = gr.Button("Run Query")

        submit_button.click(
            run_RAG_gradio_interface,
            inputs=[query, search_type],
            outputs=[result_output, context_output]
        )

        gr.Markdown("## Here you can get a random article and try to ask some questions regarding this article.")

        # Button to fetch a random article
        random_article_button = gr.Button("Get Random Article")
        random_article_title = gr.Textbox(label="Random Article Title", interactive=False)
        random_article_text = gr.Textbox(label="Random Article Text", interactive=False)

        random_article_button.click(
            random_article,
            inputs=[],
            outputs=[random_article_title, random_article_text]
        )

        # Add a title at the bottom
        gr.Markdown("### From Alina Pavliv and Oleh Lozovyi CS-415, with love <3")

    app.launch(share=True, debug=True)


if __name__ == '__main__':
    main()


#
# if __name__ == '__main__':
#     text = 'what evidence lower bound produce When used as a surrogate objective for maximum likelihood estimation in latent variable model'
#
#     print(run_RAG(text, 'vector_base')) #return answer and context files
#     # print(run_RAG(text, 'bm25')) #return answer and context files
#     # print(run_RAG(text, 'None')) #return answer, no context files
#     # print(run_RAG("WHO iS JOE BIDEN", 'bm25')) #return answer, no context files
#     # random_article() #return 1 random title with article
#
#     # search_type only None, bm25 or vector_base