import streamlit as st
import pandas as pd
import openai
import os
import random
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_data()
def load_data(file):
    """
    Load the data.
    """
    df = pd.read_csv(file, encoding="utf-8", delimiter=",")
    return pre_process(df)

def pre_process(df):
    """
    Pre-process the data.
    """
    # Drop columns that start with "Unnamed"
    for col in df.columns:
        if col.startswith("Unnamed"):
            df = df.drop(col, axis=1)
    # Replace NaN with empty string
    # df = df.fillna("")
    return df

def ask_question(question, system="You are a data scientist."):
    """
    Ask a question and return the answer.
    """ 
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question}
        ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        stop = ["plt.show()", "st.pyplot(fig)"]
        )
    answer = response.choices[0]["message"]["content"]
    return answer

def ask_question_with_retry(question, system="You are a data scientist.", retries=1):
    """
    Wrapper around ask_question that retries if it fails.
    Proactively wait for the rate limit to reset. Eg for a rate limit of 20 calls per minutes, wait for at least 2 seconds
    Compute delay using an exponential backoff, so we don't exceed the rate limit.
    """
    delay = 2 * (1 + random.random())
    time.sleep(delay)
    for i in range(retries):
        try:
            return ask_question(question, system=system)
        except Exception as e:
            delay = 2 * delay
            time.sleep(delay)
    return None

def prepare_question(description, question, initial_code):
    """
    Prepare a question for the chatbot.
    """
    return f"""
Context:
{description}
Question: {question}
Answer:
{initial_code}
"""

def describe_dataframe(df):
    """
    Describe the dataframe.
    """
    description = []
    # List the columns of the dataframe
    description.append(f"The dataframe df has the following columns: {', '.join(df.columns)}.")
    # For each column with a categorical variable, list the unique values
    if cols := check_categorical_variables(df):
        return f"ERROR: All values in a categorical variable must be strings: {', '.join(cols)}." 
    for column in df.columns:
        if df[column].dtype == "object" and len(df[column].unique()) < 10:
            description.append(f"Column {column} has the following levels: {', '.join(df[column].dropna().unique())}.")
        elif df[column].dtype == "int64" or df[column].dtype == "float64":
            description.append(f"Column {column} is a numerical variable.")
    description.append("Add a title to the plot.")
    description.append("Label the x and y axes of the plot.")
    description.append("Do not generate a new dataframe.")
    return "\n".join(description)

def check_categorical_variables(df):
    """
    Check that all values of categorical variables are strings.
    """
    # Return [] if all values of categorical variables are strings
    # Return columns if not all values of categorical variables are strings
    return [column for column in df.columns if df[column].dtype == "object" 
        and not all(isinstance(x, str) for x in df[column].dropna().unique())]

def list_non_categorical_values(df, column):
    """
    List the non-categorical values in a column.
    """
    return [x for x in df[column].unique() if not isinstance(x, str)]

def code_prefix():
    """
    Code to prefix to the visualization code.
    """
    return """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.4, 2.4))
"""
    
def generate_placeholder_question(df):
    return "Show the relationship between x and y."

def test_ask_question():
    system = "Write Python code to answer the following question. Do not include comments."
    question = "Generate a function that returns the ratio of two subsequent Fibonacci numbers."
    answer = ask_question_with_retry(question, system=system)
    print(answer)

def test_describe_dataframe():
    import pandas as pd
    df = pd.DataFrame({
        "a": ["male", "female", "male"], 
        "b": [4, 5, 6],
        "c": ["yes", "no", "yes"]})
    description = describe_dataframe(df)
    print(description)

def test_visualize_dataframe():
    import pandas as pd
    df = pd.DataFrame({
        "a": ["male", "female", "male"], 
        "b": [4, 5, 6],
        "c": ["yes", "no", "yes"]})
    question = "Show the relationship between a and c."
    description = describe_dataframe(df)
    initial_code = code_prefix()
    print(prepare_question(description, question, initial_code))

def test_visualize_dataframe_with_chat():
    import pandas as pd
    df = pd.DataFrame({
        "a": ["male", "female", "male"], 
        "b": [4, 5, 6],
        "c": ["yes", "no", "yes"]})
    question = "Show the relationship between a and c."
    description = describe_dataframe(df)
    initial_code = code_prefix()
    answer = ask_question_with_retry(prepare_question(description, question, initial_code))
    print(initial_code + answer)

st.title("Chat with your data")

uploaded_file = st.sidebar.file_uploader("Upload a dataset", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)

    with st.chat_message("assistant"):
        st.markdown("Here is a table with the data:")
        st.dataframe(df, height=200)

    question = st.chat_input(placeholder=generate_placeholder_question(df))

    if question:
        with st.chat_message("user"):
            st.markdown(question)
            
        description = describe_dataframe(df)
        if "ERROR" in description:
            with st.chat_message("assistant"):
                st.markdown(description)
        else:
            initial_code = code_prefix()
            with st.spinner("Thinking..."):
                answer = ask_question_with_retry(prepare_question(description, question, initial_code))
            with st.chat_message("assistant"):
                script = initial_code + answer + "st.pyplot(fig)"
                exec(script)
                st.markdown("Here is the code used to create the plot:")
                st.code(script, language="python")
else:
    with st.chat_message("assistant"):
        st.markdown("Upload a dataset to get started.")

# if __name__ == "__main__":    
#     # test_ask_question()
#     # test_describe_dataframe()
#     # test_visualize_dataframe()
#     test_visualize_dataframe_with_chat()