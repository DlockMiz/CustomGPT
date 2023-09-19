import sys, os, pinecone
import uuid


from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
load_dotenv()

loader = DirectoryLoader('./documents')
documents = loader.load()
embeddings = OpenAIEmbeddings()

# Load the provided documents we want to store and chunk them into appropriate sized bites
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator=".")
texts = text_splitter.split_documents(documents)

# We are only embedding the first 3 documents for demonstration purposes
texts = [texts[0], texts[1], texts[2]]
split_texts = [t.page_content for t in texts]

# Create the embedded vectors
embeds = []
for i in range(0, len(split_texts), 32):
    i_end = min(i + 32, len(split_texts))
    lines_batch = split_texts[i:i_end]
    embeds.append(embeddings.embed_documents(lines_batch))

print(embeds[0][0])


