import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import os

from PIL import Image

st.set_page_config(
    page_title="Text Summarization",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
image = Image.open("summarizer-banner.jpg")
st.image(image, caption='created by MJ')
st.title('🦜🔗 Text Summarization App')




system_openai_api_key = os.environ.get('OPENAI_API_KEY')
system_openai_api_key = st.text_input("**Step 1 :key: OpenAI Key :**", value=system_openai_api_key)
os.environ["OPENAI_API_KEY"] = system_openai_api_key



def generate_response(txt):
    # Instantiate the LLM model
    llm = OpenAI(temperature=0, openai_api_key=system_openai_api_key)
    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)


# Text input
txt_input = st.text_area('***Step 2 - Enter your text***', '', height=150)

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    # openai_api_key = st.text_input('OpenAI API Key', type = 'password', disabled=not txt_input)
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(txt_input)
            result.append(response)
            
if len(result):
    st.info(response)
