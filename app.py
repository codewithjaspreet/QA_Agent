import os
import openai
import streamlit as st
from engines import MyLLMAgent

# Streamlit App Title and Introduction
openai.api_key = os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title="Legalify.ai 🦙", page_icon="🦙", layout="wide")
openai.api_key = os.getenv("OPENAI_API_KEY")
st.title("Legalify.ai 🦙")
st.write("Unleash the power of LLMs over your data 🦙")

import streamlit as st

# User Input for Query

# Sidebar Options
engine_options = [
    'Retriever Router Engine'
    # 'Multi Step Engine',
    # 'Flare Query Engine',
    # 'Multi Doc Agent Engine',
    # 'Self Correcting Engine',
    # 'Recursive Retriever Engine'
]




selected_engine = st.sidebar.selectbox(
    'Avaliable Engine:', engine_options, index=0, help="Choose the engine for processing"
)

# Display Engine Description in Sidebar
engine_descriptions = {
    'Retriever Router Engine': """

Given a relevant context and a task in the input prompt, Large Language Models (LLMs) can effectively reason over novel information that was not observed in the training set to solve the task at hand. As a result, a popular usage mode of LLMs is to solve Question-Answering (QA) tasks over your private data. They are typically paired with a “retrieval model” to form an overall “Retrieval-Augmented Generation” (RAG) system.



""",
    'Multi Step Engine': "Engine description for Multi Step.",
    'Flare Query Engine': "Engine description for Flare Query.",
    'Multi Doc Agent Engine': "Engine description for Multi Doc Agent.",
    'Self Correcting Engine': "Engine description for Self Correcting.",
    'Recursive Retriever Engine': "Engine description for Recursive Retriever."
}

selected_engine_description = engine_descriptions.get(
    selected_engine, "No description available."
)
st.sidebar.write("Description:", selected_engine_description)


# Create an instance of MyLLMAgent
my_agent = MyLLMAgent()

query = st.chat_input("Enter your Question")


if query:

        with st.spinner("Generating Answer..."):
        # Your code here
            if selected_engine == 'Retriever Router Engine':
                print("Retriever Router Engine---Method Started")
                # my_agent.load_api_key()
                summary_dict = my_agent.load_index(query, selected_engine)
                res = my_agent.RetrieverRouterEngine(summary_dict, query)
                st.write(res)
                print('Index Loaded')


        # elif selected_engine == 'Multi Doc Agent Engine':
        #     my_agent.load_api_key()
        #     print("Multi Doc Agent Engine Method Started")
        #     res = my_agent.MultiDocAgentsEngine(query)
        #     print('Index Loaded')
        #     st.write(res)
        # # Add other engine cases as needed
        # elif selected_engine == 'Self Correcting Engine':
        #     my_agent.load_api_key()
        #     response = my_agent.SelfCorrectingEngine(query, selected_engine)
        # elif selected_engine == 'Multi Step Engine':
        #     my_agent.load_api_key()
        #     my_agent.load_index()
        #     response = my_agent.MultiStepEngine(query) 
