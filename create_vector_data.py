import pinecone

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
load_dotenv()

# Establish pinecone connection
pinecone.init(environment='us-west4-gcp-free')

# Load the provided documents we want to store and chunk them into appropriate sized bites
loader = DirectoryLoader('./documents')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator=".")
texts = text_splitter.split_documents(documents)

# Establish connection to my vector index and delete any existing vectors
index_name = "lotr"
index = pinecone.Index(index_name)
index.delete(deleteAll='true')

# Grab our default embedding model and with our chunked text create the embedded vectors and store them in pinecone.
embeddings = OpenAIEmbeddings()
Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
