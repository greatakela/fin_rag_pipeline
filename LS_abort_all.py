from dotenv import load_dotenv
load_dotenv()
from langsmith import Client
import os
import time
import random

# Get the LangSmith API key from environment variables
api_key = os.getenv("LANGSMITH_API_KEY")
project_name = os.getenv("LANGSMITH_PROJECT")

if not api_key:
    raise ValueError("LangSmith API key not found in environment variables. Please set the LANGCHAIN_API_KEY.")

# Instantiate the LangSmith Client with the API key
client = Client(api_key=api_key)

def abort_all_running_runs(project_name: str):
    """Aborts all currently executing runs in LangSmith within a specific project.

    Args:
        project_name: The name of the LangSmith project.
    """

    try:
      # Retrieve all runs with status "running" in the specified project
      runs = client.list_runs(
            status=["running"],
            project_name=project_name,
        )

      # Abort all running runs
      for run in runs:
          if run.status != "running": # Check the run status before aborting
              print(f"Skipping run with ID: {run.id} because its status is {run.status}")
              continue

          max_retries = 3
          retry_delay = 1 # seconds

          for attempt in range(max_retries):
            try:
                client.update_run(run_id=run.id, status="aborted")
                print(f"Successfully aborted run with ID: {run.id}")
                break # Exit retry loop if successful

            except Exception as e:
              if "Conflict" in str(e) or "payload already received" in str(e):
                print(f"Conflict error for run ID: {run.id}, retrying in {retry_delay} seconds... Attempt {attempt + 1}/{max_retries}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

              else:
                print(f"Failed to abort run with ID: {run.id}. Error: {e}")
                break # Exit retry loop for non-conflict errors

          else:
            print(f"Failed to abort run with ID: {run.id} after {max_retries} attempts.")

      print("Finished processing all running LangSmith runs.")

    except Exception as e:
        print(f"Failed to list running runs. Error: {e}")


if __name__ == "__main__":
    # Replace "your_project_name" with the actual project name in LangSmith
    project_name_to_abort = project_name
    abort_all_running_runs(project_name_to_abort)
