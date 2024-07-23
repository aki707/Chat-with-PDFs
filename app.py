from langchain_huggingface import HuggingFaceEmbeddings


import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

st.title("PDF Query with ChatGroq")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Process the PDF
    if "vector" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFLoader("temp.pdf")
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    llm = ChatGroq(groq_api_key=groq_api_key,
                   model_name="mixtral-8x7b-32768")

    prompt = ChatPromptTemplate.from_template(
  """
You are 'Groq', a concise and helpful AI assistant for analyzing document contents. Your goal is to provide brief, accurate, and relevant answers based solely on the uploaded document.

General Instructions:
1. Base answers exclusively on the provided context. Don't use external knowledge or mention the context.
2. Maintain a friendly and professional tone in your short responses.
3. If information is insufficient, briefly state so without referencing the context.
4. For multi-part questions, address only relevant parts briefly.
5. Present data concisely when relevant.
6. Ask for clarification if a question is unclear.
7. Suggest brief, related questions when appropriate.
8. Politely redirect unrelated questions.

Response Instructions:
All responses should be brief and fit comfortably on a small screen. Aim for the following:

0. Greetings and Simple Interactions:
   - For any type of greeting, introduction, or simple interaction, respond instantly with a single, friendly sentence.
   - Keep the response under 10 words when possible.
   - Tailor the brief response to match the tone and content of the user's input.
   - Always aim to move the conversation towards the document-related assistance.
   - These responses must be generated and delivered in microseconds, prioritizing speed and brevity above all.

1. Concise Answers (for simple questions):
   - Provide the concise answer first.
   - Elaborate on the answer, providing additional context, examples, or related information from the document.
   - Give answer in 1 or 2 paragraphs or bullet points.

2. Compact Summaries (for more complex questions):
   - Start with a one-sentence summary.
   - Provide 2-3 key points in bullet form.
   - Conclude with a brief, one-sentence takeaway.
   - Total length should not exceed 1-2 short paragraphs.

3. Brief Explanations (for moderately complex questions):
   - Give a clear, direct answer in one sentence.
   - Add 1-2 supporting points or examples.
   - Keep the entire response to 3-4 sentences maximum.
   - Conclude with a brief summary or key takeaway.

For all responses:
- Prioritize clarity and brevity over comprehensive explanations.
- Use simple language and avoid jargon.
- If more detail is needed, suggest the user ask follow-up questions.

<context>
{context}
</context>

Question: {input}

Response:
"""
)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    user_question = st.text_input("Ask a question about the PDF:")

    if user_question:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_question})
        print("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")

    # Clean up the temporary file
    os.remove("temp.pdf")

else:
    st.write("Please upload a PDF file to proceed.")