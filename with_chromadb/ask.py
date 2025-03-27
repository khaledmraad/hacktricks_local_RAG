import chromadb
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.prompts.chat import ChatPromptTemplate
from pathlib import Path

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Initialize the chromadb client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Create or get the collection
collection = chroma_client.get_or_create_collection(name="growing_vegetables")

# Get the user's query
user_query = input("What do you want to know about growing vegetables?\n\n")

# Query the collection
results = collection.query(
    query_texts=[user_query],
    n_results=1
)

# Initialize the LLM (Ollama)
llm = Ollama(model="llama2")

# Define the system prompt
system_prompt = """
You are a helpful assistant. You answer questions about growing vegetables in Florida. 
But you only answer based on knowledge I'm providing you. You don't use your internal 
knowledge and you don't make things up.
If you don't know the answer, just say: I don't know and stop generating responses.
--------------------
The data:
""" + str(results['documents']) + """
"""

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{user_query}")  # Pass the user_query variable here
    ]
)

# Build the LLM chain
llm_chain = prompt_template | llm

# Display the response
print("\n\n---------------------\n\n")
response = llm_chain.invoke({"user_query": user_query})  # Use user_query, not country

print(response)

# Debug: Check the actual message content
print("done")
# print(response.choices[0].message.content)
