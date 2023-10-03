# About
This app allows you to explore a dataset by asking questions about it. It uses the OpenAI API to generate code to answer your questions and visualize the results.

This is a rewrite from the ground up of the [Chat2VIS](https://github.com/frog-land/Chat2VIS_Streamlit) app.

# Dependencies
The app uses streamlit, pandas, and matplotlib.

You need an OpenAI API key to use this app. You can get one [here](https://beta.openai.com/). You also need to add it your environment variables as OPENAI_API_KEY. 

# How to use
Start the app by using the following command:

```
streamlit run chat_data.py
```

Upload a dataset you want to examine. 

Then, ask start questions of the data. For example, if your dataset contains information about companies, their countries of origin, and funding received, you can ask:

- Show the average funding by company.
- Show the number of companies by country.
- Show the total funding by country.

The app will generate code to answer your questions. If the code executes successfully, it will also generate a plot. You can then copy and paste the code.

Sometimes, you need to tweak your question to get the code to work properly.