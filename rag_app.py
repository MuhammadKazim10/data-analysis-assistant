import streamlit as st
import pandas as pd
import io
from langchain.llms import OpenAI
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import sys
from io import StringIO
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.error("OpenAI API is not set. Kindly set it in .env file.")

# Function to process the uploaded CSV file
def process_csv(file):
    return pd.read_csv(file)

# Function to generate Python code based on the question
def generate_code(df, question):
    llm = OpenAI(temperature=0.7)
    
    # Generate detailed column information
    column_info = ", ".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
    df_info = f"Columns and types: {column_info}\nShape: {df.shape}"
    
    prompt = PromptTemplate(
        input_variables=["df_info", "question"],
        template="""
        You are a data analysis assistant. Given a pandas DataFrame with the following information:
        {df_info}
        
        Generate Python code to answer the following question about the data:
        {question}

        Instructions:
        1. Do not read the dataset or make any imports. The DataFrame is already imported as 'df', and pandas and matplotlib are imported as 'pd' and 'plt', respectively.
        2. Use the column names and data types provided to write specific code for this dataset.
        3. Avoid unnecessary checks except verifying column existence and type appropriateness.
        4. If a query cannot be executed, write code to print a reason why it cannot be executed.
        5. Use print statements for all outputs. If the output spans multiple lines (e.g., value counts in a column), ensure each line of output is distinct and clear.
        6. For visual outputs, use matplotlib to create and show plots.
        7. Use '.items()' instead of '.iteritems()' for iterating over Pandas Series objects.
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(df_info=df_info, question=question)
    return response.strip()

# Function to safely execute the generated code
def execute_code(code, df):
    local_vars = {"df": df, "pd": pd, "plt": plt}
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    
    try:
        exec(code, globals(), local_vars)
        sys.stdout = old_stdout
        return redirected_output.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error: {str(e)}"

# Streamlit app
def main():
    st.title("Data Query Assistant")

    if 'df' not in st.session_state:
        st.session_state.df = None

    uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")
    if uploaded_file is not None:
        st.session_state.df = process_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")
        st.write(st.session_state.df.head())

    if st.session_state.df is not None:
        st.subheader("Ask a question about your data")
        question = st.text_input("Enter your question here:")
        if st.button("Get Answer"):
            if question:
                with st.spinner("Generating and executing code..."):
                    generated_code = generate_code(st.session_state.df, question)
                    st.subheader("Generated Python Code:")
                    st.code(generated_code, language="python")
                    
                    result = execute_code(generated_code, st.session_state.df)
                    st.subheader("Result:")
                    st.write(result)
                    
                    # Check if the code includes a plot
                    if "plt." in generated_code:
                        st.pyplot(plt)  # Display the plot

if __name__ == "__main__":
    main()