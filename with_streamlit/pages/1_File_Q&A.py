import streamlit as st
import anthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from langchain_community.llms import Ollama
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"


llm = Ollama(model="llama2")

st.title("📝 File Q&A with Anthropic")
uploaded_file = st.file_uploader("Upload an article",type="pdf")
try: 
    b = uploaded_file.getvalue()
    # print(uploaded_file.type) doesnt matter
    with open("./data/"+uploaded_file.name, "wb") as f:
        f.write(b)
except:
    pass



question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
)

def fill_db(DATA_PATH_arg):
    loader = PyPDFDirectoryLoader(DATA_PATH_arg)

    raw_documents = loader.load()

    # splitting the document

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(raw_documents)

    # preparing to be added in chromadb

    documents = []
    metadata = []
    ids = []

    i = 0

    for chunk in chunks:
        documents.append(chunk.page_content)
        ids.append("ID"+str(i))
        metadata.append(chunk.metadata)

        i += 1

    # adding to chromadb


    collection.upsert(
        documents=documents,
        metadatas=metadata,
        ids=ids
    )


def reemovNestings(l):
    for i in l:
        if type(i) == list:
            reemovNestings(i)
        else:
            output.append(i)


if uploaded_file and question :
    output = []

    print(question)



    # Initialize the chromadb client
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name="tmp")
    
    fill_db(DATA_PATH)

    results = collection.query(
        query_texts=[question],
        n_results=2
    )

    reemovNestings(results['documents'])
    
    query_output_string = " ".join(output)
    print  (query_output_string)
    system_prompt = """
    You are a RAG agent that answers questions about a PDF from a retrieved document. 
    But you only answer based on knowledge I'm providing you. You don't use your internal 
    knowledge and you don't make things up.
    If you don't know the answer, just say: I don't know and stop generating responses.
    --------------------
    The data:
    """ + query_output_string + """
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
    response = llm_chain.invoke({"user_query": question})  # Use user_query, not country

    

    # article = uploaded_file.read().decode()
    # print(article)

    st.write("### Answer")
    st.write(response)
    # chroma_client.delete_collection(name="tmp")