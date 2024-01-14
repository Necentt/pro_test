import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import (
    load_tools,
    initialize_agent,
    AgentType
)
import langchain
import gradio as gr
import matplotlib
from dotenv import load_dotenv


load_dotenv('../.env')
matplotlib.use('TkAgg')

langchain.debug = True
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
chat = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo", verbose=True)

tools = load_tools([], llm=chat)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(
    tools,
    chat,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors="Check your output and make sure it conforms!",
    memory=memory
)


def call_agent(user_question):
    response = agent.run(input=user_question)
    return response


with gr.Blocks() as demo:
    title = gr.HTML("<h1>mememmemememememe</h1>")
    input = gr.Textbox(label="Че хочешь узнать про мемы?")
    output = gr.Textbox(label="Держи ответ братишка")
    btn = gr.Button("ЧЕ????")
    btn.click(fn=call_agent, inputs=input, outputs=output)


demo.launch(share=True, debug=True)