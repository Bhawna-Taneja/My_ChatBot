import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import TensorflowHubEmbeddings
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
def get_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000,chunk_overlap=200,length_function=len)
    text_chunks = splitter.split_text(text)
    return text_chunks
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding=embeddings)
    return vectorstore
def get_convo(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)    
    convo_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory = memory
    )
    return convo_chain
    
def handle_userinput(user_ques):
    response = st.session_state.conversation({'question':user_ques})
    #st.write(response)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
def main():
    load_dotenv()
    st.set_page_config(page_title="ChatBot by Bhawna", page_icon=":book:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("My_ChatBot")
    user_ques=st.text_input("Ask a Question from your Doc:")
    if user_ques:
        handle_userinput(user_ques)

    with st.sidebar:
        st.subheader("Your doc")
        pdf_docs = st.file_uploader("Upload your pdfs and click on Submit",accept_multiple_files=True)
        
        if st.button("Submit"):
            with st.spinner("Processing"):
                #get pdf text
                raw_text = get_text(pdf_docs)
                #st.write(raw_text)
                #get text chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)
                
                #create knowlege base(vector store)
                vectorstore = get_vectorstore(text_chunks)
                
                # create coversation chain
                st.session_state.conversation = get_convo(vectorstore)
                
if __name__ == '__main__':
    main()