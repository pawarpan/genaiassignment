import streamlit as st
import pandas as pd
import time
from sklearn.impute import SimpleImputer
import numpy as np
import pickle as pt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 
import base64 



analysis_rows=[]


if "page" not in st.session_state:
    st.session_state.page = "Analysis"

if "df" not in st.session_state:
    st.session_state.df = None

def switch_page(page_name):
    st.session_state.page = page_name
    st.rerun()

if "run_regression" not in st.session_state:
    st.session_state.run_regression = False


def main():
    st.title("Regression Analysis App")
    analysis_page()

def analysis_page():
    st.header("Agentic AI Sprint Analysis Tool", divider=True)
    st.write("This project is a part of the course EPGDPMAI at IIM Indore.")
    st.subheader("Team Members:", divider=True)
    st.write("- Pankaj Pawar (EPGDPMAI/B3/2024/13)")
    st.write("- Saranjit Singh (EPGDPMAI/B3/2024/13)")
    st.subheader("", divider=True)
    st.write("Please upload your dataset in CSV,txt or xlsx format.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv","txt","xlsx"])
    separator = st.text_input("Enter separator (default is ',')", value=",")
    
    st.subheader("Connect to ChatGPT for live Summary", divider=True)
    connect_gpt = st.checkbox("Connect to ChatGPT for live Summary ??", key="connect_gpt")

    gpt_key=""
    if connect_gpt:
        gpt_key = st.text_input("Enter your GPT API Key", type="password", key="gpt_key")


    if st.button("Confirm and Proceed"):
        st.write("You can now proceed with the analysis.")
        


    if uploaded_file is not None:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            st.session_state["df"] = df
        else:
            df = pd.read_csv(uploaded_file, sep=separator)
        st.write("File uploaded successfully!")
        st.write("Data Preview:")
        st.dataframe(df.head())
        st.subheader("Data Description:" , divider=True)    
        st.dataframe(df.describe())
        st.subheader("" , divider=True) 



        run_regression_clicked = st.button("Start Agentic AI process and Analysis")

        if run_regression_clicked:
            st.session_state.run_regression = True

        if st.session_state.run_regression:
                    st.write("Connecting to Agentic AI for analysis...")
                    with st.spinner("Running Analysis..."):
                        performAnalysis(gpt_key,df)


def plot_model_error():
        st.subheader("Model Performance Comparison", divider=True)
        fig, ax = plt.subplots()
        sns.barplot(x=[row[0] for row in analysis_rows], y=[row[1] for row in analysis_rows], ax=ax)
        ax.set_title("Model Performance (MSE)")
        ax.set_xlabel("Regression Models")
        ax.set_ylabel("Mean Squared Error (MSE)")
        ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels by 45 degrees
        st.pyplot(fig)


def perform_preprocessing(df,features,target_variable):
    st.header("Performing preprocessing...", divider=True)
    
    X = df[features]
    Y = df[target_variable]
    
    st.write("features and target variable")

    st.markdown("###")
    st.write("**Independent Variables (X):**")
    for feature in features:
        st.write(f"- {feature}")
   
    st.markdown("###")
    st.write("**Dependent Variables (X):**")
    st.write(f"- {target_variable}")
   
    combined_df = pd.DataFrame(X, columns=features)
    combined_df[target_variable] = Y.values

    # Highlight the target variable and features in the DataFrame
    def highlight_columns(column):
        if column in features:
            return 'background-color: red; color: white;'
        elif column == target_variable:
            return 'background-color: blue; color: white;'
        return ''

    styled_df = combined_df.head().style.applymap(highlight_columns, subset=combined_df.columns)
    st.write(combined_df)
    
    X= X.iloc[:,:].values
    Y= Y.iloc[:,:].values
      
    # Check for categorical columns and apply SimpleImputer
    categorical_columns = np.array([not np.issubdtype(dtype, np.number) for dtype in df[features].dtypes])
    categorical_indices = np.where(categorical_columns)[0]
    st.write("Categorical Column Names:")
    st.write(df[features].columns[categorical_columns])
    #st.write(categorical_indices)
    #st.write(type(categorical_indices))
    #st.write(categorical_columns.size)

    if categorical_indices.size > 0:
        st.subheader("Handling Missing Values for Categorical Columns:",  divider=True)
        st.write(df[features].columns[categorical_columns])

        imputer = SimpleImputer(strategy="most_frequent")
        X[:, categorical_indices] = imputer.fit_transform(X[:, categorical_indices])
        st.write("Missing values in categorical columns have been imputed with the most frequent value.")
        for i, col in enumerate(df[features].columns[categorical_columns]):
            st.write(f"Imputed value for column  **{col}**: **{imputer.statistics_[i]}**")

        st.write("Independent Variables (X) after categorical imputation:")
        st.dataframe(X)
    else:
        st.write("No categorical columns found for imputation.")
    



    # Check for numerical columns and apply SimpleImputer
    numerical_columns = np.array([np.issubdtype(dtype, np.number) for dtype in df[features].dtypes])
    numerical_indices = np.where(numerical_columns)[0]

    if numerical_columns.size > 0:
        st.subheader("Handling Missing Values for Numerical Columns:",  divider=True)
        #st.write(numerical_indices)
        imputer = SimpleImputer(strategy="mean")
        X[:, numerical_columns] = imputer.fit_transform(X[:, numerical_columns])
        
        st.write("Imputed values for numerical columns:")
        imputed_data = {
            "Column Name": df[features].columns[numerical_columns],
            "Imputed Value": imputer.statistics_
        }
        imputed_df = pd.DataFrame(imputed_data)
        st.dataframe(imputed_df)
    else:
        st.write("No numerical columns found for imputation.")

#----

    # Check for categorical columns and apply SimpleImputer
    st.subheader("Handling Missing Values for Categorical Columns of Dependent Variable:",  divider=True)
    
    categorical_columns_y = np.array([not np.issubdtype(dtype, np.number) for dtype in df[target_variable].dtypes])
    categorical_indices_y = np.where(categorical_columns_y)[0]
    
    #categorical_columns_y = not np.issubdtype(df[target_variable].dtype, np.number)
    #categorical_indices_y = [0] if categorical_columns_y else []
    if categorical_columns_y > 0:
        #st.write(df[target_variable].columns[categorical_columns_y])
        imputer = SimpleImputer(strategy="most_frequent")
        X[:, categorical_indices_y] = imputer.fit_transform(X[:, categorical_indices_y])
        st.write("Missing values in categorical columns have been imputed with the most frequent value.")
        for i, col in enumerate(df[target_variable].columns[categorical_columns_y]):
            st.write(f"Imputed value for column '{col}': {imputer.statistics_[i]}")

        st.write("Dependent Variables (Y) after categorical imputation:")
        st.dataframe(X)
    else:
        st.write("No categorical columns found for imputation for Dependent Variable.")
    



    # Check for numerical columns and apply SimpleImputer
    st.subheader("Handling Missing Values for Numerical Columns of Dependent Variable:",  divider=True)

    numerical_columns_y = np.array([np.issubdtype(dtype, np.number) for dtype in df[target_variable].dtypes])
    numerical_indices_y = np.where(numerical_columns)[0]

    numerical_columns_y = np.array([np.issubdtype(dtype, np.number) for dtype in df[target_variable].dtypes])
    #st.write(numerical_columns_y)
    #st.write(numerical_indices_y[0])
    st.write(Y)

    print("print type")
    print(type(X))
    print(X)
    print(type(Y))
    print(Y)

    if numerical_columns_y:
        imputer = SimpleImputer(strategy="mean")
        Y[:, numerical_columns_y] = imputer.fit_transform(Y[:, numerical_columns_y])
        
        st.write("Imputed values for numerical columns:")
        imputed_data = {
            "Column Name": df[target_variable].columns[numerical_columns_y],
            "Imputed Value": imputer.statistics_
        }
        imputed_df = pd.DataFrame(imputed_data)
        st.dataframe(imputed_df)
    else:
        st.write("No numerical columns found for imputation.")

#-


    st.write("Handling Missing Data Completed!") 
    st.write("Independent Variables (X) after handling Missing Data :")
    st.dataframe(X)  # Display first 5 rows of X as a dataframe

    #handle Categorical data.
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    if categorical_indices.size > 0:
        st.subheader("Encoding Categorical Data...", divider=True)        
        # Perform Label Encoding
        label_encoder = LabelEncoder()
        for index in categorical_indices:
            X[:, index] = label_encoder.fit_transform(X[:, index])
       
        st.write("Categorical data has been encoded using LabelEncoder.")
        st.write("Independent Variables (X) after Label Encoding:")
        st.dataframe(X)
        
        # Perform OneHotEncoding
        column_transformer = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(), categorical_indices)
            ],
            remainder='passthrough'
        )
        X = column_transformer.fit_transform(X)
        st.write("Categorical data has been encoded using OneHotEncoder.")
        st.write("Independent Variables (X) after OneHotEncoding:")
        st.dataframe(X)
        st.write("Encoding Categorical Data Completed!")
    else:
        st.write("No categorical columns found for encoding.")



    st.subheader("Applying StandardScaler to normalize the data...", divider=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    st.write("Data after applying StandardScaler:")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Independent Variables (X):")
        st.dataframe(X)
    with col2:
        st.write("Dependent Variable (Y):")
        st.dataframe(Y)
    st.write("Data Preprocessing Completed!")
    return X, Y

def performAnalysis(gpt_key,df):
    st.write("Starting Analysis...")
    analyze_correlation_with_gpt(gpt_key, df,  reply="")




def chat_with_gpt(api_key ,df , input_text):
    import openai
    print("Welcome to ChatGPT CLI (type 'exit' to quit)")
    openai.api_key=api_key
    conversation = []
    
    print("Inside Chatp GPT")
    user_input = input_text

    # Convert the dataframe to a string (e.g., CSV format) to pass as context
    df_string = df.to_csv(index=False)

    # Add the dataframe as a system message to provide context to ChatGPT
    conversation.append({
        "role": "system",
        "content": f"The following is the data you should use for answering questions:\n{df_string}"
    })
    conversation.append({"role": "user", "content": user_input})
    try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Or use "gpt-3.5-turbo" if you want a cheaper option
                messages=conversation
            )
            reply = response.choices[0].message['content'].strip()
            conversation.append({"role": "assistant", "content": reply})
            print("ChatGPT:", reply)
            return reply
    except Exception as e:
            print("Error:", e)


def analyze_correlation_with_gpt(gpt_key, dataforgpt, reply , isimage=False):

            import pandas as pd
            if isinstance(dataforgpt, pd.DataFrame):
                dataforgpt_str = dataforgpt.to_csv(index=False)
            else:
                dataforgpt_str = str(dataforgpt)


            st.subheader("ChatGPT Output:", divider=True)
            reply=""
            with st.spinner("Connecting to ChatGPT and analyzing"):
                reply = chat_with_gpt(gpt_key, dataforgpt, "you are an expert scrum master and you have to do the analysis of the sprint wise data provided. Provide a detailed analysis of the data including correlation analysis, insights, and recommendations for improvement based on the data provided. Be concise and to the point.")
            st.subheader("ChatGPT Response:", divider=True)
            st.write(reply)
            st.subheader("", divider=True)

            with st.spinner("Connecting to ChatGPT and further analysis"):
                reply = chat_with_gpt(gpt_key, dataforgpt, "Based on the previous analysis, can you provide specific recommendations for improving the sprint performance? Focus on actionable insights that can help the team enhance their productivity and efficiency.please suggest differenct charts and metrics to visualize the sprint performance and track progress over time. here is the previous reply : " + reply)
            st.subheader("ChatGPT Response:", divider=True)
            st.write(reply)


            df= dataforgpt
            st.subheader("Sprint Velocity Trend")
            velocity = df.groupby("Sprint Name")["Story Points"].sum()
            st.line_chart(velocity)

            st.subheader("Story Points Delivered per Sprint")
            st.bar_chart(velocity)

            with st.spinner("Connecting to ChatGPT and further analysis"):
                reply = chat_with_gpt(gpt_key, dataforgpt, "based on sprint velocity trend provide a detailed analysis of the sprint velocity trend. Identify any patterns, fluctuations, or anomalies in the velocity over time. Additionally, suggest strategies to maintain a consistent and sustainable velocity for future sprints.")
            st.subheader("ChatGPT Response:", divider=True)
            st.write(reply)




            st.subheader("Contributor Productivity")
            assignee_prod = df.groupby("Assignee")["Story Points"].sum()
            st.bar_chart(assignee_prod)

            with st.spinner("Connecting to ChatGPT and further analysis"):
                reply = chat_with_gpt(gpt_key, dataforgpt, "based on assigne wise productivity provide a detailed analysis of the contributor productivity. Identify top performers, areas for improvement, and strategies to enhance overall team productivity. Additionally, suggest ways to recognize and reward high-performing contributors.")
            st.subheader("ChatGPT Response:", divider=True)
            st.write(reply)


            st.subheader("Role-wise Productivity")
            role_prod = df.groupby("Role")["Story Points"].sum()
            st.bar_chart(role_prod)

            with st.spinner("Connecting to ChatGPT and further analysis"):
                reply = chat_with_gpt(gpt_key, dataforgpt, "based on role wise productivity provide a detailed analysis of the role-wise productivity. Identify which roles are contributing the most to story points and which roles may need additional support or resources. Additionally, suggest strategies to optimize role allocation and enhance overall team performance.")
            st.subheader("ChatGPT Response:", divider=True)
            st.write(reply)



            st.subheader("Status Distribution")
            status_counts = df["Status"].value_counts()
            st.bar_chart(status_counts)


            with st.spinner("Connecting to ChatGPT and further analysis"):
                reply = chat_with_gpt(gpt_key, dataforgpt, "based on status distribution provide a detailed analysis of the status distribution. Identify any bottlenecks or areas where tasks are getting stuck. Additionally, suggest strategies to improve workflow efficiency and ensure timely completion of tasks.")
            st.subheader("ChatGPT Response:", divider=True)
            st.write(reply)


            st.subheader("Copilot Usage vs Productivity")
            copilot_prod = df.groupby("Copilot Usage")["Story Points"].mean()
            st.bar_chart(copilot_prod)


            with st.spinner("Connecting to ChatGPT and further analysis"):
                reply = chat_with_gpt(gpt_key, dataforgpt, "based on copilot usage vs productivity provide a detailed analysis of the impact of copilot usage on productivity. Identify any correlations between copilot usage and story points delivered. Additionally, suggest strategies to optimize copilot usage for enhancing team productivity.")
            st.subheader("ChatGPT Response:", divider=True)
            st.write(reply)

            st.subheader("Sprint-wise Status Distribution")
            sprint_status = df.groupby(["Sprint Name", "Status"]).size().unstack(fill_value=0)
            st.bar_chart(sprint_status)

            with st.spinner("Connecting to ChatGPT and further analysis"):
                reply = chat_with_gpt(gpt_key, dataforgpt, "based on sprint-wise status distribution provide a detailed analysis of the sprint-wise status distribution. Identify any trends or patterns in task statuses across different sprints. Additionally, suggest strategies to improve task management and ensure a balanced distribution of work throughout the sprints.")
            st.subheader("ChatGPT Response:", divider=True)
            st.write(reply)


            st.subheader("Delay Reason Analysis")
            delay_counts = df["Delay Reason"].value_counts()
            st.bar_chart(delay_counts)

            with st.spinner("Connecting to ChatGPT and further analysis"):
                reply = chat_with_gpt(gpt_key, dataforgpt, "based on delay reason analysis provide a detailed analysis of the reasons for delays in task completion. Identify the most common delay reasons and their impact on overall sprint performance. Additionally, suggest strategies to mitigate these delays and improve task completion rates.")
            st.subheader("ChatGPT Response:", divider=True)
            st.write(reply)


if st.session_state.df is not None:
    df = st.session_state.df



exec_context = {
    "st": st,
    "df": st.session_state["df"],
    "pd": pd
}

import ast

def validate_generated_code(code: str):
    tree = ast.parse(code)
    banned = {"exec", "eval", "open", "os", "sys", "subprocess"}

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in node.names:
                if name.name.split(".")[0] in banned:
                    raise ValueError("Unsafe import detected")

    return True


# Load and encode the image
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def  chat_with_gpt_image(api_key ,img , input_text):
    import openai
    # Encode plot image
    image_base64 = encode_image_to_base64(img)

    # Call OpenAI with image
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # GPT-4 with vision
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Please summarize the plot shown in the image."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }}
            ]}
        ],
        max_tokens=500
    )

    return response.choices[0].message['content']

main()