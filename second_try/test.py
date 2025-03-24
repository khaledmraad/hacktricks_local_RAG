from requests import post as rpost
from langchain_core.language_models.llms import LLM
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import HumanMessage
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)



def call_llama(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama2",
        "prompt": prompt,
        "stream": False,
    }

    response = rpost(
        "http://localhost:11434/api/generate",
        headers=headers,
        json=payload
    )
    return response.json()["response"]


class LLaMa(LLM):
    def _call(self, prompt, **kwargs):
        return call_llama(prompt)

    @property
    def _llm_type(self):
        return "llama2"

documents = [
    {"content": "What is your return policy? ..."},
    {"content": "How long does shipping take? ..."},
    # Add more documents as needed
]

texts = [doc["content"] for doc in documents]

retriever = FAISS.from_texts(
    texts,
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
).as_retriever(k=5)

faq_template = """
You are a chat agent for my E-Commerce Company. As a chat agent, it is your duty to help the human with their inquiry and make them a happy customer.

Help them, using the following context:
<context>
{context}
</context>
"""

faq_prompt = ChatPromptTemplate.from_messages([
    ("system", faq_template),
    MessagesPlaceholder("messages")
])


document_chain = create_stuff_documents_chain(LLaMa(), faq_prompt)

def parse_retriever_input(params):
    return params["messages"][-1].content

retrieval_chain = RunnablePassthrough.assign(
    context=parse_retriever_input | retriever
).assign(answer=document_chain)


from langchain.schema import HumanMessage

response = retrieval_chain.invoke({
    "messages": [
        HumanMessage("I received a damaged item. I want my money back.")
    ]
})

print(response)

