import os
import io
import traceback
import pandas as pd
from typing import List, Dict, Any
import copy
import re 

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load .env first
load_dotenv()

# Assuming your LangGraph module is in 'app.langgraph_app'
# NOTE: Ensure you have an 'app' directory with 'langgraph_app.py' inside it,
# or adjust the import path.
from app.langgraph_app import run_langgraph_query

# --- CONFIGURATION ---
MOCK_MODE = os.getenv("MOCK_MODE", "False").lower() == "true"
DATA_PATH = os.getenv("DATA_PATH")
DESCRIPTION_PATH = os.getenv("DESCRIPTION_PATH")
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 10))

if not DATA_PATH or not DESCRIPTION_PATH:
    raise ValueError("DATA_PATH or DESCRIPTION_PATH environment variable not set. Please check your .env file.")


# --- DYNAMIC DATA AND DESCRIPTION LOADING ---
def load_data_from_path(data_path: str) -> pd.DataFrame:
    """Loads data exclusively from a public Google Sheet URL."""
    # Check if the path is a valid Google Sheets URL
    if data_path.startswith("http") and "docs.google.com/spreadsheets" in data_path:
        try:
            # 1. Extract the main Sheet ID
            sheet_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', data_path)
            if not sheet_id_match:
                raise ValueError("Invalid Google Sheet URL format: Could not find the main Sheet ID.")
            sheet_id = sheet_id_match.group(1)

            # 2. Extract the specific Sheet Tab ID (gid). If not found, default to gid=0.
            gid_match = re.search(r'gid=(\d+)', data_path)
            # Use gid=0 if no specific GID is found in the URL.
            gid = gid_match.group(1) if gid_match else '0'

            # 3. Construct the public CSV export URL
            # NOTE: The sheet must be publicly shared ("Anyone with the link can view").
            csv_export_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}'
            
            # 4. Read directly into a DataFrame
            df = pd.read_csv(csv_export_url)
            print(f"Successfully loaded data from Google Sheet with ID {sheet_id} and GID {gid}")
            return df
        
        except Exception as e:
            # Catch errors during the access/loading process
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load data from {data_path}. Ensure the Google Sheet is publicly shared. Error: {e}"
            )
    else:
        # Strictly reject non-Google Sheet paths
        raise HTTPException(
            status_code=400,
            detail="DATA_PATH must be a Google Sheets sharing URL. Local CSV files are no longer supported in this configuration."
        )

# Load data and description once at server startup
try:
    df = load_data_from_path(DATA_PATH)
    with open(DESCRIPTION_PATH, "r", encoding="utf-8") as f:
        data_description = f.read()
except HTTPException as e:
    # Re-raise the HTTPException to stop the server from starting if data fails to load
    raise e
except Exception as e:
    raise FileNotFoundError(f"Failed to load description file: {DESCRIPTION_PATH}. Error: {e}")


app = FastAPI(title="Easy Insights Backend")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class QueryPayload(BaseModel):
    user_query: str
    history: List[Dict[str, Any]]

# --- THE MAIN QUERY ENDPOINT ---

@app.post("/query")
async def query(payload: QueryPayload):
    """The main endpoint for conversational analytics."""

    # 1. Prepare history (Remove large objects for LLM context)
    cleaned_history = []
    for turn in payload.history:
        cleaned_turn = copy.deepcopy(turn)
        # Ensure 'answer' key exists before attempting to delete
        if 'answer' in cleaned_turn:
            cleaned_answer = cleaned_turn['answer']
            if 'formatted_table' in cleaned_answer:
                del cleaned_answer['formatted_table']
            if 'visualization' in cleaned_answer:
                del cleaned_answer['visualization']
        cleaned_history.append(cleaned_turn)

    print(f"User query: {payload.user_query}")

    # 2. Run Query
    try:
        # The LangGraph function receives the global 'df' and 'data_description' loaded at startup
        if MOCK_MODE:
            # (MOCK MODE LOGIC - KEPT FOR REFERENCE)
            sql_query = "SELECT col1, SUM(col2) FROM df GROUP BY col1;"
            db_response = df.head().to_dict(orient="records")
            final_response = "Mock mode: Analysis is running, but showing mock data."
            return {
                "final_state": {
                    "question": payload.user_query,
                    "sql_query": sql_query,
                    "db_response": db_response,
                    "final_response": final_response,
                }
            }
        else:
            # NOTE: Your run_langgraph_query must accept df and data_description
            result = run_langgraph_query(payload.user_query, df, data_description, cleaned_history)
            return {"final_state": result}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"final_state": {"final_response": f"Internal Server Error: {str(e)}", "insights": "An error occurred during analysis."}}
        )