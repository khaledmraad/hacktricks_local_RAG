from langchain_community.llms import Ollama
import streamlit as st
from langchain.prompts.chat import ChatPromptTemplate


st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by Ollama")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Define the system prompt
system_prompt = """
You are a helpful assistant.
"""




if prompt := st.chat_input():

    # Define the prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{user_query}")  # Pass the user_query variable here
        ]
    )


    print(prompt)
    # Build the LLM chain
    llm = Ollama(model="llama2")

    llm_chain = prompt_template | llm

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = llm_chain.invoke({"user_query": prompt}) 
    msg = response
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
