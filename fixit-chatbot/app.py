
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from flask import Flask
from flask import Flask, render_template, request



os.environ["OPENAI_API_KEY"] = "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"

import textract
doc = textract.process("/home/shubhamraj/Desktop/fixit/Lucidity_merged.pdf")

with open('prd.txt', 'w') as f:
    f.write(doc.decode('utf-8'))

with open('prd.txt', 'r') as f:
    text = f.read()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

type(chunks[0])
token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

df = pd.DataFrame({'Token Count': token_counts})

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(chunks, embeddings)

chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

chat_history = []


app = Flask(__name__)

@app.route("/chatbot")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    result = qa({"question": userText, "chat_history": chat_history})
    response= result['answer']
    return response
app.run(debug = True)

"""
print("Welcome to the lucidity chatbot! Type 'exit' to stop.")
while True:
    query = input("Enter a query !")

    if query.lower() == 'exit':
        print("Thank you for using Lucidity Chat Bot!")
        break

    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    print(chat_history)
"""



