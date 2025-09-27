import streamlit as st
import requests
import json
import hashlib
import os
import base64
import pandas as pd
import time
import re # Import the regex module for stripping the index

# ------------------------
# Config
# ------------------------
BACKEND_URL = "http://127.0.0.1:8000/query"
HISTORY_FILE = "chat_history.json"

st.set_page_config(page_title="Easy Insights — Conversational Analytics", layout="wide")

# Custom CSS for enhanced table and UI styling
st.markdown("""
<style>
    .stDataFrame {
        font-family: Arial, Calibri, sans-serif;
    }
    .stDataFrame .css-9s5dg .css-1cpx6h0 th {
        background-color: #f0f2f6;
        color: black;
        font-weight: bold;
        text-transform: uppercase;
        border-bottom: 2px solid #ccc;
    }
    .stDataFrame .css-9s5dg .css-1cpx6h0 td {
        background-color: white;
        color: black;
        border-bottom: 1px solid #eee;
    }
    .stDataFrame .css-9s5dg .css-1cpx6h0 .dataframe-row-1 {
        background-color: #f8f2f6;
    }
    .stDataFrame .css-9s5dg .css-1cpx6h0 .header {
        font-size: 1.1em;
    }
    .stDataFrame table {
        border-collapse: collapse;
    }
    .stDataFrame th, .stDataFrame td {
        border-right: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------
# Helpers
# ------------------------
def make_key(base: str, idx: int, turn_idx: int):
    """Generate unique button keys using md5 of text + index + turn index."""
    return f"follow_{turn_idx}_{idx}_{hashlib.md5(base.encode()).hexdigest()}"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def stream_text(text, delay=0.01):
    """Yields text word by word."""
    words = text.split()
    for i in range(len(words)):
        yield " ".join(words[:i+1])
        time.sleep(delay)

# ------------------------
# Init session state
# ------------------------
if "history" not in st.session_state:
    st.session_state.history = load_history()

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# NEW: Counter for resetting the text_input key after submission/execution
if "input_key_counter" not in st.session_state:
    st.session_state.input_key_counter = 0

# ------------------------
# Query Submission Function (Original logic preserved)
# ------------------------
def handle_query_submission(user_query):
    if not user_query:
        return # Do nothing if query is empty

    try:
        # Create an empty container for the progress bar
        progress_bar = st.progress(0, text="Starting analysis...")
        
        # Step 1: Connecting to backend
        progress_bar.progress(20, text="1. Connecting to backend...")
        time.sleep(0.5)

        # The requests.post call will block until the backend responds
        response = requests.post(
            BACKEND_URL,
            json={"user_query": user_query, "history": st.session_state.history}
        )

        if response.status_code == 200:
            result = response.json().get("final_state", {})
            
            # Step 2: Generating code and executing
            progress_bar.progress(50, text="2. Generating and executing analysis code...")
            time.sleep(1) # Simulate delay

            # Step 3: Generating final insights and a summary
            progress_bar.progress(80, text="3. Generating final insights and a summary...")
            time.sleep(1) # Simulate delay

            st.session_state.history.append({
                "question": user_query,
                "answer": result
            })
            save_history(st.session_state.history)
            
            # CRITICAL: Increment the key counter to force the text input to reset on rerun
            st.session_state.input_key_counter += 1
            
            # Step 4: Complete
            progress_bar.progress(100, text="Done! Displaying results.")
            time.sleep(0.5)
            progress_bar.empty() # Remove the progress bar
            st.rerun() # Rerun to display the new history
        else:
            progress_bar.empty()
            st.error(f"Backend error: {response.status_code}")
    except Exception as e:
        progress_bar.empty()
        st.error(f"Error contacting backend: {str(e)}")


# ------------------------
# UI
# ------------------------
st.title("Easy Insights - Chat (Demo)")
st.markdown("Ask questions about your dataset")

# ------------------------
# Query Log Sidebar (FIXED DOUBLE INDEXING by stripping internal index)
# ------------------------
with st.sidebar:
    st.header("Query Log")
    if not st.session_state.history:
        st.info("No queries yet.")
    else:
        for idx, turn in enumerate(st.session_state.history):
            query_text = turn['question']
            
            # FIX: Remove any leading index (e.g., "1. " or "2. ") that the backend might have added.
            # Regex pattern to match "1." or "1." followed by optional space at the beginning.
            clean_query = re.sub(r'^\s*\d+\.\s*', '', query_text).strip()
            
            # Now, display the correct index followed by the clean query.
            st.markdown(f"**{idx + 1}.** {clean_query}")

# ------------------------
# Display history
# ------------------------
for idx, turn in enumerate(st.session_state.history):
    st.markdown(f"🧑 **You:** {turn['question']}")
    data = turn["answer"]

    if not data:
        continue

    # Assistant answer - dynamic streaming
    if "final_response" in data:
        response_container = st.empty()
        # Check if the chat turn has been fully rendered
        if not turn.get("fully_rendered", False):
            full_response = ""
            for word in data["final_response"].split():
                full_response += word + " "
                response_container.markdown(f"🤖 **Assistant:** {full_response.strip()}")
                time.sleep(0.05)
            # Mark this turn as fully rendered
            turn["fully_rendered"] = True
            response_container.markdown(f"🤖 **Assistant:** {data['final_response']}")
        else:
            # If already rendered, just display the full text
            st.markdown(f"🤖 **Assistant:** {data['final_response']}")

    # Results table and chart in a two-column layout
    if "formatted_table" in data and data["formatted_table"]:
        cols = st.columns(2)
        with cols[0]:
            st.markdown("### **Table**")
            try:
                # Convert the list of dicts back to a DataFrame for display
                df_to_display = pd.DataFrame(data["formatted_table"])
                st.dataframe(df_to_display, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying table: {e}")

        with cols[1]:
            if "visualization" in data and data["visualization"]:
                st.markdown("### **Chart**")
                try:
                    # Decode the base64 image and display it
                    image_bytes = base64.b64decode(data["visualization"])
                    st.image(image_bytes, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying visualization: {e}")

    # Display insights
    if "insights" in data:
        st.markdown("### **Insights**")
        st.markdown("##### Executive Summary")
        st.write(data["insights"])

    # Display follow-up questions in a horizontal layout
    if "follow_ups" in data and data["follow_ups"]:
        st.markdown("### **Suggested Follow-ups**")
        follow_up_cols = st.columns(len(data["follow_ups"]))
        for i, q in enumerate(data["follow_ups"]):
            with follow_up_cols[i]:
                if st.button(q, key=make_key(q, i, idx)):
                    # CRITICAL FIX: Directly call the submission logic with the follow-up question (q)
                    handle_query_submission(q)

    st.markdown("---")

# ------------------------
# Main input section 
# ------------------------

# 1. Capture the value for this run (set by a follow-up button on the previous run, if any)
input_value_for_run = st.session_state.user_input

# 2. Immediately clear the user_input state variable. This ensures the text box is empty 
# unless a follow-up button sets the value for the current run.
st.session_state.user_input = ""

user_query = st.text_input(
    "Ask a question:",
    value=input_value_for_run,
    # 3. CRITICAL: Use the unique key to force the widget to reset its internal state (clearing the text box)
    key=f"user_input_box_{st.session_state.input_key_counter}" 
)

# Submit button logic
if st.button("Submit Query"):
    # Call the reusable function with the current query from the text box
    handle_query_submission(user_query)


# Clear chat button
if st.button("Clear Chat"):
    st.session_state.history = []
    st.session_state.user_input = ""
    # CRITICAL: Increment key to ensure a fresh, empty text box
    st.session_state.input_key_counter += 1 
    save_history([])
    st.rerun()