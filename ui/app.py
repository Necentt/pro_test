import gradio as gr
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.schema import AIMessage, HumanMessage


template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

prompt = PromptTemplate(template=template, input_variables=["question"])
model_path = "../models/openchat_3.5.Q4_K_M.gguf"
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.0,
    max_tokens=2190,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)



