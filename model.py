from langchain.chains.question_answering.map_rerank_prompt import output_parser
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import streamlit as st
import PyPDF2

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY','Your Api Key')
os.environ['LANGCHAIN_TRACKING_V2'] = 'true'

def read_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()+'\n'

    return text

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a good assistant.Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

uploader = st.file_uploader("Upload Research Paper:")

model = Ollama(model = 'llama2')
output = StrOutputParser()
chain = prompt|model|output

if uploader:
    content = read_pdf(uploader)
    model.invoke(content)
    st.title("Question:")
    input_text = st.text_input("Search the topic u want")
    st.write(chain.invoke({"question":input_text}))

















