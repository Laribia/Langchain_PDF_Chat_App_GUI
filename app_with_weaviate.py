import os
import json
import pandas as pd
import streamlit as st
import pickle
from dotenv import load_dotenv
import weaviate
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.chains import VectorDBQA
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStoreRetriever

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

def load_read_extract(pdf_file):
    try:
        # Save uploaded file to tmp_folder
        tmp_folder = os.path.join("tmp", pdf_file.name)
        with open(tmp_folder, "wb") as tmp_file:
            tmp_file.write(pdf_file.getbuffer())
        
        # Load and read uploaded pdf
        pdf_loader = PyPDFLoader(tmp_folder)
        pdf_doc = pdf_loader.load()
    except Exception:
        print('No PDF Document was uploaded !!')
    return pdf_doc

def jprint(json_in):
    print(json.dumps(json_in, indent=2))

def main():
    st.subheader(':speech_balloon: Chat with your PDF Documents :speech_balloon:')
    
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    
    # Upload PDF Document
    pdf_file = st.file_uploader("Upload your PDF Document here :point_down:", type='pdf')
    if pdf_file is not None:
        pdf_doc = load_read_extract(pdf_file)    
    
        if pdf_doc:
            # Chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=50,
                length_function=len
            )
            chunks = text_splitter.split_documents(pdf_doc)
            chunks_content = [chunk.page_content for chunk in chunks]

            store_name = pdf_file.name[:-4]
            # Create a dictionary with the data
            # data = {'text': chunks_content}
            data = {'text': chunks_content}
            data_df = pd.DataFrame(data)
            
            # st.write(embeddings_df.head())
            # # Anything uploaded to weaviate is automatically persistent into the database.
            # Import custom embedding vectors
            client = weaviate.Client(
                url=os.environ["WEAVIATE_URL"],
                auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"]),
                additional_headers={
                    "X-OpenAI-API-Key": os.environ["OPENAI_API_KEY"]
                }
            )
            client.schema.delete_all()
            pdf_class = {
                "class": "PDF",
                "vectorizer": "text2vec-openai",
                "moduleConfig": {
                    "text2vec-openai": {
                        "model": "ada",
                        "modelVersion": "001",
                        "vectorizeClassName": True
                    }
                }
            }
            client.schema.create_class(pdf_class)
            st.write(client.schema.get())
            client.batch.configure(batch_size=len(chunks))
            with client.batch as batch:
                for i in range(len(data_df)):
                    pdf_obj = {
                        "text" :data_df["text"].iloc[i]
                    }
                    batch.add_data_object(pdf_obj, "PDF")
            
            # st.write(client.query.aggregate(pdf_class["class"]).with_meta_count().do())
            vector_store = Weaviate(client=client, index_name="PDF", text_key="text")
                    
            # # Get user questions/queries
            query = st.text_input("Now that you have uploaded your PDF document,\
                                you can ask you questions about it:")
            if query:
                retriever = vector_store.as_retriever(search_kwargs={"k":3})
                # cands = retriever.get_relevant_documents(query)
                # VectorStoreRetriever(client)
                llm = ChatOpenAI()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm, chain_type="stuff", retriever=retriever
                )
                with get_openai_callback() as cb:
                    response = qa_chain(query, return_only_outputs=True)
                    print(cb)
                st.write(response)

        
if __name__ == '__main__':
    main()