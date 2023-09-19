import sys, pinecone

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone

from dotenv import load_dotenv
load_dotenv()

# Grab the given text
query = sys.argv[1]

# Establish which open ai model to use and grab the default embedding. 
# We can also change the temperature of the model which dictates the accurarcy of the response, 0 being the most accurate 1 being the least. The drawback being the higher accuracy results in higher response times.
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
embeddings = OpenAIEmbeddings()

# Initialize pinecone connection and get the vectorstore based on our embedding model. Then with the vectorstore, take our query and retrieve the most similar vectors.
pinecone.init(environment='us-west4-gcp-free')
index = pinecone.Index('lotr')
vectorstore = Pinecone(index, embeddings.embed_query, text_key='text')
found_pinecone_docs = vectorstore.similarity_search(query)

# Print the found documents
# print(found_pinecone_docs)

# Establish a question and answer chain with the given llm and with the found pinecone docs, submit our query.
chain = load_qa_chain(llm, chain_type='stuff')
print(chain.run(input_documents=found_pinecone_docs, question=query))
