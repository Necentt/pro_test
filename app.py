from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
import gradio as gr
import time
import matplotlib
matplotlib.use('TkAgg')


code_llama_model = LlamaCpp(
    model_path="models/openchat_3.5.Q4_K_M.gguf",
    config={'max_new_tokens': 512, 'temperature': 0.0}
)

# prompt_template = 'Ты умный ассистент, отвечающий на вопросы пользователя о файлах, содержащихся в базе данных. '\
#                   'Полезные для ответа части текста из файлов будут поданы в контексте. '\
#                   'Используй их, чтобы дать точный и полный ответ на вопрос пользователя. '\
#                   'Не матерись. Все матерные слова заменяй на звездочки. '
prompt_template = "You are a helpful, respectful and honest assistant. " \
                  "Always answer as helpfully as possible, while being safe.  " \
                  "Your answers should not include any harmful, unethical, racist, " \
                  "sexist, toxic, dangerous, or illegal content. Please ensure" \
                  " that your responses are socially unbiased and positive in nature. " \
                  "If a question does not make any sense, or is not factually coherent, " \
                  "explain why instead of answering something not correct. If you don't know " \
                  "the answer to a question, please don't share false information."

with gr.Blocks(title='PRO_DEMO') as demo:
    chatbot = gr.Chatbot([], elem_id="Chatbot", height=500)
    user_input = gr.Textbox()
    clear_button = gr.ClearButton([user_input, chatbot])


    def generate_response(query):
        prompt = PromptTemplate(template=prompt_template, input_variables=['query'])
        chain = LLMChain(prompt=prompt, llm=code_llama_model)
        response = chain.run({'query': query})
        return response


    def chat_with_bot(message, chat_history):
        bot_message = generate_response(message)
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history


    user_input.submit(chat_with_bot, [user_input, chatbot], [user_input, chatbot])


demo.launch()
