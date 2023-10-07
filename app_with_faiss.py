import os
import streamlit as st
import pickle
from dotenv import load_dotenv
import chromadb
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
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

load_dotenv()

def main():
    st.subheader(':speech_balloon: Chat with your PDF Documents :speech_balloon:')
    
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
            chunk_size=5000,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_documents(pdf_doc)
        # st.write(chunks)
        # embeddings
        store_name = pdf_file.name[:-4]
        st.write(store_name)
        embeddings = OpenAIEmbeddings(model="gpt-3.5-turbo")
        # persistent_directory = f"db_{store_name}"
        # if os.path.exists(persistent_directory):
        #     vector_store = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
               vector_store = pickle.load(f)
        else:
            vector_store = FAISS.from_documents(chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)
        # vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persistent_directory)
        # Get user questions/queries
        query = st.text_input("Now that you have uploaded your PDF document,\
                              you can ask you questions about it:")
        
        if query:
            docs = vector_store.similarity_search(query=query)
            # st.write(docs)
            llm = ChatOpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

        # if query:
        #     retriever = vector_store.as_retriever(search_kwargs={"k":3})
        #     llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        #     qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        #     with get_openai_callback() as cb:
        #         response = qa_chain.run(query)
        #         print(cb)
        #     st.write(response)

        
if __name__ == '__main__':
    main()