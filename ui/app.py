import gradio as gr
import matplotlib
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv


load_dotenv('../.env')
matplotlib.use('TkAgg')
openai_api_key = os.getenv('OPENAI_API_KEY')
chat = ChatOpenAI(
    openai_api_key=openai_api_key,
    model='gpt-3.5-turbo'
)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

CHROMA_PATH = "../chroma"


def main(query_text):

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


with gr.Blocks() as demo:
    title = gr.HTML("<h1>mememmemememememe</h1>")
    example = 'Кто такие Сигмы?'
    input1 = gr.Textbox(label="Спроси про известные мемы 2023 года", value=example)
    output = gr.Textbox(label="Ответ")
    btn = gr.Button(value='Получить ответ')

    btn.click(fn=main, inputs=input1, outputs=output)
    print(input1)


demo.launch(share=True, debug=True)