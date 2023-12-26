import streamlit as st
from engines import MyLLMAgent

# Streamlit App Title and Introduction
st.title("Legalify.ai 🦙")
st.write("Unleash the power of LLMs over your data 🦙")

# User Input for Query

# Sidebar Options
engine_options = [
    'Retriever Router Engine',
    'Multi Step Engine',
    'Flare Query Engine',
    'Multi Doc Agent Engine',
    'Self Correcting Engine',
    'Recursive Retriever Engine'
]

selected_engine = st.sidebar.selectbox(
    'Select an Engine:', engine_options, index=0, help="Choose the engine for processing"
)

# Display Engine Description in Sidebar
engine_descriptions = {
    'Retriever Router Engine': "Engine description for Retriever Router.",
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

query = st.text_input("Enter your Question")

if st.button("Execute Query"):
    if selected_engine == 'Retriever Router Engine':
        print("Retriever Router Engine---Method Started")
        summary_dict = my_agent.load_index(query, selected_engine)
        res = my_agent.RetrieverRouterEngine(summary_dict, query)
        st.write(res)
        print('Index Loaded')
    elif selected_engine == 'Multi Doc Agent Engine':
        print("Multi Doc Agent Engine Method Started")
        res = my_agent.MultiDocAgentsEngine(query)
        print('Index Loaded')
        st.write(res)
    # Add other engine cases as needed
    elif selected_engine == 'Self Correcting Engine':
        response = my_agent.SelfCorrectingEngine(query, selected_engine)
