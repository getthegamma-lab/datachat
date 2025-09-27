import os
import io
import base64
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import json
import re 

from openai import OpenAI

# Load API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# Intent Classifier
# -------------------------
def classify_user_intent(user_query: str, history=None):
    """
    Classifies the user's intent to determine if it's a data query or a conversational response.
    """
    intent_prompt = f"""
    You are a conversational AI assistant for a data analytics application. Your primary job is to answer questions about a dataset by generating Python code.
    However, sometimes a user's question is not a data query but a conversational follow-up to a previous answer.

    Analyze the user's query and the conversation history to classify the intent.

    User query: "{user_query}"
    Conversation history:
    {json.dumps(history, indent=2)}

    Is this a request for a new data analysis (e.g., "show me total sales", "what is the trend of revenue") or a conversational follow-up (e.g., "why is that happening?", "tell me more about that", "This is interesting.")?

    Respond with ONLY a single word: "DATA" or "CONVERSATIONAL".
    """
    
    intent_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": intent_prompt}],
        temperature=0.0
    )
    
    return intent_resp.choices[0].message.content.strip().upper()

# -------------------------
# Core Query Runner
# -------------------------
def run_langgraph_query(user_query: str, df: pd.DataFrame, data_description: str, history=None):
    """
    Runs query through LLM and ensures consistent structured output by executing generated code.
    """

    try:
        # Step 1: Classify user intent
        intent = classify_user_intent(user_query, history)

        if intent == "CONVERSATIONAL":
            # Handle conversational follow-ups
            context_prompt = f"""
            Based on the following conversation history, provide a concise and helpful conversational response to the user's latest query. Do not perform any new data analysis.

            Conversation history:
            {json.dumps(history, indent=2)}

            Latest user query: {user_query}

            Assistant's response:
            """
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": context_prompt}]
            )
            return {
                "question": user_query,
                "final_response": response.choices[0].message.content.strip(),
                "formatted_table": None,
                "visualization": None,
                "insights": None,
                "follow_ups": [],
            }
        
        # If intent is DATA, proceed with the existing workflow
        # Step 2: Perform Exploratory Data Analysis (EDA) to get a deep understanding of the data
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_string = buffer.getvalue()
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_summary = ""
        for col in categorical_cols:
            if df[col].nunique() < 20: # Limit to columns with a reasonable number of unique values
                categorical_summary += f"- Top 5 values in '{col}':\n{df[col].value_counts().head().to_string()}\n\n"

        data_summary = f"""
        DATASET SCHEMA AND STATISTICAL SUMMARY:
        - Columns and their data types:
        {info_string}
        
        - Statistical summary of numerical columns:
        {df.describe(include='all').to_string()}
        
        - Key insights from a quick look at the data:
        {categorical_summary}
        """

        # Step 3: Combine the EDA with the existing data description for a single, rich prompt
        full_data_context = f"""
        DATASET DESCRIPTION:
        {data_description}
        
        ---
        
        {data_summary}
        """

        # Step 4: Prompt the LLM to generate a Python code block for analysis and a chart
        code_prompt = f"""
        You are a senior data analyst. Based on the following user query and a comprehensive summary of the dataset, generate a single, self-contained Python code block.
        
        Dataset Context:
        {full_data_context}

        User query: {user_query}

        Instructions:
        1. **Code Block**: The code must perform the requested analysis on the 'df' DataFrame.
        2. **STRICTLY PROHIBITED**: DO NOT use `import os`, `import sys`, `import subprocess`, `import multiprocessing`, or any file/system manipulation functions. The only allowed imports are pandas, matplotlib, seaborn, base64, io, and plotly.
        3. **Output Variables**: The code must output two variables:
           - `formatted_table`: A pandas DataFrame containing the tabular results.
           - `visualization`: A base64-encoded image string of a relevant chart. Use `matplotlib.pyplot` or `seaborn` for plots. Save the figure to a buffer, encode it, and then close the plot to prevent memory leaks.
        4. **Specific Requirements**:
           - **Percentage Column**: For any aggregation or summary table, calculate the percentage of the total for the primary numerical column and add it as a new column named '% of Total'.
           - **Sorting**: Sort the final table by the primary numerical column in descending order.
           - **Chart**: For all categorical aggregations (e.g., sales by product), use a **bar chart** as the primary visualization. **Ensure the chart is readable.** Rotate x-axis labels if they are long and use `plt.tight_layout()` to prevent labels from overlapping or being cut off.
        5. **Do not use any external files**. All operations must be in memory.

        Example of desired output structure:
        ```python
        # Code block starts here
        # Your python code here...
        # Example:
        # result_df = df.groupby('category').agg(total_sales=('sales', 'sum')).reset_index()
        # total = result_df['total_sales'].sum()
        # result_df['% of Total'] = (result_df['total_sales'] / total * 100).round(2).astype(str) + '%'
        # result_df = result_df.sort_values(by='total_sales', ascending=False)
        # formatted_table = result_df
        # fig, ax = plt.subplots(figsize=(10, 6)) # Example of adjusting size
        # sns.barplot(x='category', y='total_sales', data=result_df, order=result_df['category'])
        # plt.xticks(rotation=45, ha='right') # Rotate and align labels
        # plt.tight_layout() # Adjust plot to fit rotated labels
        # img_buffer = io.BytesIO()
        # fig.savefig(img_buffer, format='png')
        # img_buffer.seek(0)
        # visualization = base64.b64encode(img_buffer.read()).decode('utf-8')
        # plt.close(fig)
        # End of your code
        ```
        """
        code_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": code_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        
        # Step 5: Extract and execute the generated Python code block
        full_response_text = code_response.choices[0].message.content.strip()
        code_block_match = re.search(r"```python(.*?)```", full_response_text, re.DOTALL)
        if not code_block_match:
            # Added a fallback if no code block is found to still give a conversational answer
            response_text = full_response_text
            if response_text and len(response_text) > 50:
                 return {
                    "question": user_query,
                    "final_response": response_text,
                    "formatted_table": None,
                    "visualization": None,
                    "insights": "The model provided a conversational response instead of code. Trying again.",
                    "follow_ups": [],
                }
            raise ValueError("No Python code block found in the LLM's response.")
            
        code_block = code_block_match.group(1).strip()

        # 🛡️ CRITICAL FIX: Safety check for common problematic imports/commands
        problematic_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'from\s+multiprocessing', 
            r'os\.(system|popen|startfile)',
            r'subprocess\.',
            r'shutil\.',
        ]

        for pattern in problematic_patterns:
            if re.search(pattern, code_block):
                raise ValueError("Generated code contained disallowed system or OS commands. Aborting execution.")

        local_env = {
            "df": df,
            "pd": pd,
            "plt": plt,
            "io": io,
            "base64": base64,
            "px": px,
            "sns": sns,
            "formatted_table": None,
            "visualization": None
        }
        
        # ✅ FIX #1: Removed '{"__builtins__": None}' to allow standard built-ins for internal system calls (multiprocessing fix)
        exec(code_block, {}, local_env)
        
        formatted_table = local_env.get("formatted_table", None)
        visualization = local_env.get("visualization", None)

        # Step 6: Generate insights based on keywords and table data
        user_query_lower = user_query.lower()
        is_striking = False
        
        multidimensional_keywords = ['by', 'trend', 'comparison', 'breakdown', 'group', 'compare', 'across']
        for keyword in multidimensional_keywords:
            if keyword in user_query_lower:
                is_striking = True
                break

        if is_striking:
            insights_prompt = f"""
            Based on the user's query: "{user_query}" and the following tabular data:
            {formatted_table.to_string() if formatted_table is not None else "No table"}
            
            Act as a senior business consultant providing a comprehensive executive summary.
            Your insights must be:
            1. **High-Level Summary**: Start with a concise statement about the overall top performers. For example, "Latte is the top-selling coffee type..."
            2. **Detailed Narrative**: Provide a thorough analysis of trends, patterns, and anomalies from the table.
            3. **Actionable**: Suggest next steps or business implications based on the findings.
            4. **Quantified**: Use specific numbers and percentages from the data to support your points.
            5. **Structured**: Use bullet points.
            """
        else:
            insights_prompt = f"""
            Based on the user's query: "{user_query}" and the following tabular data:
            {formatted_table.to_string() if formatted_table is not None else "No table"}
            
            As a business consultant, provide a very concise, one-to-two-sentence executive summary. **Do not state the obvious from the table.** Focus on a key takeaway or business implication. Do not use bullet points.
            """
        
        insights_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": insights_prompt}]
        )
        insights = insights_resp.choices[0].message.content.strip()

        # Step 7: Generate relevant follow-up questions based on the data summary and the results
        fu_prompt = f"""
        Based on the user's query: "{user_query}"
        The dataset summary is:
        {data_summary}
        The results from the analysis are:
        {formatted_table.to_string() if formatted_table is not None else "No table"}
        
        Suggest 3-4 smart, relevant follow-up questions the user could ask. **These questions must be answerable with the data provided.** Do not ask about data that does not exist in the dataset.
        """
        fu_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a proactive data consultant."},
                      {"role": "user", "content": fu_prompt}]
        )
        follow_ups = [q.strip("•- \n") for q in fu_resp.choices[0].message.content.split("\n") if q.strip()]

        # ✅ FIX #2: Removed .fillna(value=None) as it is unnecessary and throws a ValueError.
        # pandas .to_dict() already handles NaNs by converting them to JSON 'null'.
        return {
            "question": user_query,
            "final_response": "Analysis Summary: ",
            "formatted_table": formatted_table.to_dict(orient="records") if formatted_table is not None else None,
            "visualization": visualization,
            "insights": insights,
            "follow_ups": follow_ups,
        }

    except Exception as e:
        traceback.print_exc()
        error_message = f"An error occurred during analysis. Error: {str(e)}"
        
        if "pandas.errors.ParserError" in str(e):
            error_message = "An error occurred while parsing the data. Please ensure the data file is a valid CSV."
        elif "No Python code block found" in str(e):
            error_message = "I was unable to generate the required Python code to perform this analysis. Please try rephrasing your query."
        elif "'seaborn'" in str(e) or "'matplotlib'" in str(e):
            error_message = f"An error occurred: The required visualization library is not installed. Please add it to your `requirements.txt` file and run `pip install -r requirements.txt`."
        elif "disallowed system or OS commands" in str(e):
            error_message = "The generated code attempted to use a disallowed system command. I will try again with a cleaner prompt on the next request."


        return {
            "question": user_query,
            "final_response": error_message,
            "formatted_table": None,
            "visualization": None,
            "insights": "I was unable to process this request. Please try rephrasing your query or check for missing libraries.",
            "follow_ups": [],
        }