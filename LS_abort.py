from dotenv import load_dotenv
load_dotenv()
from langsmith import Client
import os


# Get the LangSmith API key from environment variables
api_key = os.getenv("LANGSMITH_API_KEY")

if not api_key:
    raise ValueError("LangSmith API key not found in environment variables. Please set the LANGCHAIN_API_KEY.")


# Instantiate the LangSmith Client with the API key
client = Client(api_key=api_key)


def stop_langsmith_run(run_id: str):
    """Stops a LangSmith run by setting its status to 'aborted'.

    Args:
      run_id: The ID of the LangSmith run to stop.
    """

    try:
      client.update_run(run_id=run_id, status="aborted")
      print(f"Successfully aborted run with ID: {run_id}")
    except Exception as e:
        print(f"Failed to abort run with ID: {run_id}. Error: {e}")


if __name__ == "__main__":
    # Replace "your_run_id" with the actual run ID you want to stop
    run_id_to_stop = "297991f2-8350-41c3-8aec-f80f8fc905b7"
    stop_langsmith_run(run_id_to_stop)
