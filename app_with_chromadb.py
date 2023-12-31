import os
import streamlit as st
import pickle
from dotenv import load_dotenv, find_dotenv, set_key
import chromadb
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.chains import VectorDBQA
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown(
        """
        ## **About**
        :blue[This app is an LLM-powered chatbot To ask you PDF documents built using:]
        - [Streamlit](https://docs.streamlit.io.) 
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) :gray[LLM model]
    

        """
    
    )
    add_vertical_space(5)
    st.write(':call_me_hand:*:orange[Made by **Mouad Laribia**]*:wink:')

def get_api_key():
    t1 = st.tabs(['OpenAI API key'])
    openai_api_key = st.text_input(
        'OPENAI_API_KEY',
        type='password',
        key='OPENAI_API_KEY',
        # on_change=,
        label_visibility="collapsed"
    )

    return openai_api_key

# # Set openai api key
# openai_api_key = get_api_key()
# if openai_api_key is not None:
#     set_key(".env", key_to_set="OPENAI_API_KEY", value_to_set=openai_api_key, quote_mode="always")
    
## Get the key value
# load_dotenv()

def store_embeddings(path, document_chunks, embeddings):
    if os.path.exists(path) and 'index' in os.listdir(path):
        vector_store = Chroma(persist_directory=path, embedding_function=embeddings)
    else:
        vector_store = Chroma.from_documents(document_chunks, embeddings, path)
    vector_store.persist()
    return vector_store

def ask_me(query, vector_store, openai_api_key, model_name='gpt-4'):
    retriever = vector_store.as_retriever(search_kwargs={"k":4})
    cands = retriever.get_relevant_documents(query)
    # st.write(cands)
    llm = ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    # qa_chain = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=vector_store)
    # cands = qa_chain._get_docs(query)
    with get_openai_callback() as cb:
        response = qa_chain.run(query)
        print(cb)
    return response

def main():
    st.subheader(':speech_balloon: Chat with your PDF Documents :speech_balloon:')
    # st.write('Enter your OpenAI API key')
    openai_api_key = get_api_key()
    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    # Upload PDF Document
    pdf_file = st.file_uploader("Upload your PDF Document here :point_down:", type='pdf')
    
    if pdf_file is not None:
        tmp_folder = os.path.join("tmp", pdf_file.name)
        with open(tmp_folder, "wb") as tmp_file:
            tmp_file.write(pdf_file.getbuffer())
        
        pdf_loader = PyPDFLoader(tmp_folder)
        pdf_doc = pdf_loader.load()
        
        # chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_documents(pdf_doc)
        
        store_name = pdf_file.name[:-4]
        st.write(store_name)

        # embeddings
        # openai_api_key = os.getenv("OPENAI_API_KEY")
        # print(openai_api_key)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        persistent_directory = os.path.join("db", f"db_{store_name}")
        vector_store = store_embeddings(persistent_directory, chunks, embeddings)
        
        # Get user questions/queries
        query = st.text_input("Now that you have uploaded your PDF document,\
                              you can ask you questions about it:")
        if query:
            response = ask_me(query, vector_store, openai_api_key)
            st.write(response)

if __name__ == '__main__':
    main()
    