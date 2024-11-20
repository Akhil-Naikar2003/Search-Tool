import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = st.secrets['GROQ_API_KEY']

# Setup Wikipedia query
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# Setup search query
search = DuckDuckGoSearchRun(name="Search")

# Setup Streamlit app
st.title("🔎 Mni Search Engine - Chat with Search")
"""
Stuck with anything or want to know more about something? Feel free to ask!
"""

# Sidebar settings
st.sidebar.title("Settings")
with st.sidebar:
    if st.button("Return to Main Menu"):
            st.markdown("[Go Back](http//:192.168.142.1:3000/T2)", unsafe_allow_html=True)
    st.write("""
    1. Ask questions in the chat input box.\n
    2. Use the app for web search via DuckDuckGo.\n
    3. Get word meanings from Wikipedia.\n
    4. Ask for detailed information on topics.\n
    5. Get real-time updates and responses.\n
    6. Use the assistant for casual and technical queries.
    """)

# User input for new question
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.chat_message("user").write(prompt)

    # Initialize language model and tools only once per query
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, wiki]
    
    # Initialize search agent with error handling for parsing
    search_agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        handle_parsing_errors=True  # This helps the agent handle errors
    )

    # Handle response generation
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = search_agent.run([{"role": "user", "content": prompt}], callbacks=[st_cb])
            st.write(response)
        except Exception as e:
            st.write(f"An error occurred: {str(e)}")
