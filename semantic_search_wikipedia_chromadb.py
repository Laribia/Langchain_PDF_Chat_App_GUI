import os
import torch
import chromadb
import pandas as pd
from dotenv import load_dotenv
from typing import List
import wikipedia
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

def store_embeddings(path, document_chunks, embeddings):
    if os.path.exists(path) and 'index' in os.listdir(path):
        vector_store = Chroma(persist_directory=path, embedding_function=embeddings)
    else:
        vector_store = Chroma.from_documents(document_chunks, embeddings, path)
    vector_store.persist()
    return vector_store

import wikipedia
from typing import List
from collections import defaultdict

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

candidates = ["Rab Butler", "Robin Butler, Baron Butler of Brockwell"]

text_splitter = CharacterTextSplitter(chunk_size=1500)

doc = defaultdict(list)
documents: List[Document] = []

for candidate in candidates:
    page = wikipedia.page(title=candidate, auto_suggest=False)
    text = page.content
    chunk_text = text_splitter.split_text(text)
    # doc[page.title] = chunk_text
    for chunk in chunk_text:
        documents.append(Document(page_content=chunk, metadata={"name": page.title}))

# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(documents, embeddings, persist_directory="db")
query=f"""from text above, which name in {candidates} refers to 'Butler'?

text = '''
Butler launches attack on Blair. Former civil service chief Lord Butler has criticised the way Tony Blair's government operates, accusing it of being obsessed with headlines.
He also attacked the way the Iraq war was "sold" to the public, with important warnings on the strength of the intelligence left out. Tory leader Michael Howard said Lord Butler had given the "most damaging testimony" he could remember. But Downing Street said Mr Blair should be judged by results not his style.
Lord Butler said Mr Blair bypassed the Cabinet and relied instead on small, informal groups of advisers to help him make decisions. The prime minister's official spokesman said the Cabinet was still used to achieve a consensus on important issues. But he added: "You cannot, in a modern government, take every decision in Cabinet. It's just not possible."
Lord Butler said the government had too much freedom to "bring in bad Bills" and "to do whatever it likes" and it relied too much on the advice of political appointees. The former cabinet secretary said in an interview with The Spectator magazine: "I would be critical of the present government in that there is too much emphasis on selling, there is too much central control and there is too little of what I would describe as reasoned deliberation which brings in all the arguments." Mr Howard described Lord Butler's intervention as "very important". "This is from someone who was an insider at the very heart of the Blair government. "It is certainly the most damaging testimony I can ever remember from someone in such an eminent position."
Lord Butler's report earlier this year into Iraq intelligence said the government's September 2002 weapons dossier did not make clear intelligence about claims that Saddam Hussein had stockpiles of chemical and biological weapons was "very thin". The reason for this is that it would have weakened ministers' case for war, Lord Butler said in his Spectator interview, which was conducted by the magazine's editor, Conservative MP Boris Johnson. He said: "When civil servants give material to ministers, they say these are the conclusions we've drawn, but we've got to tell you the evidence we've got is pretty thin. "Similarly, if you are giving something to the United Nations and the country you should warn them."
Asked why he thought the warnings were not there Lord Butler said: "One has got to remember what the purpose of the dossier was. The purpose of the dossier was to persuade the British why the government thought Iraq was a very serious threat." When asked whether he thought the country was well-governed on the whole, he replied: "Well. I think we are a country where we suffer very badly from Parliament not having sufficient control over the executive, and that is a very grave flaw. "We should be breaking away from the party whip. The executive is much too free to bring in a huge number of extremely bad Bills, a huge amount of regulation and to do whatever it likes - and whatever it likes is what will get the best headlines tomorrow. "All that is part of what is bad government in this country." Lord Butler's assessment was backed by his predecessor as Cabinet Secretary, Lord Armstrong. Lord Armstrong told BBC Two's Newsnight: "I agree ... there doesn't appear to be the sort of informed collective political judgement brought to bear on decision-making that those affected by decisions are entitled to expect." Liberal Democrat deputy leader Menzies Campbell said he thought Lord Butler's comments were "well justified" and Mr Blair's style of leadership was "corrosive of the whole system of government". But Labour former minister Jack Cunningham accused Lord Butler of basing his comments on the first eight months of the incoming Labour administration, when he was cabinet secretary. Mr Cunningham told BBC Radio 4's Today programme: "Taken together, Robin Butler's comments are partial, inaccurate and cannot be taken as anything other than politically biased against the Labour government." ''' """

similarities_scores = vector_store.similarity_search_with_score(
    query=text,
    k=3
)

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name='gpt-4'),
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

with get_openai_callback() as cb:
    response = qa(query)
    print(cb)