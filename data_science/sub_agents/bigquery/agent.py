# Modified based on the code from 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Database Agent: get data from database (BigQuery) using NL2SQL."""

import os

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from . import tools
from .chase_sql import chase_db_tools
from .prompts import return_instructions_bigquery

NL2SQL_METHOD = os.getenv("NL2SQL_METHOD", "BASELINE")

def setup_before_agent_call(callback_context: CallbackContext) -> None:
    """Setup the agent."""

    # Retrieve the 'database_settings' from the callback_context.state.
    # This 'database_settings' is set by the root agent's setup,
    # and it now contains the 'bq_ddl_schemas' dictionary.
    database_settings = callback_context.state.get("database_settings", {})

    # Check if 'bq_ddl_schemas' exists in database_settings
    if "bq_ddl_schemas" not in database_settings:
        raise ValueError("'bq_ddl_schemas' not found in database_settings. Ensure tools.update_database_settings is called and populates it correctly.")

    all_ddl_schemas = database_settings["bq_ddl_schemas"]
    dataset_ids = list(all_ddl_schemas.keys())

    if not dataset_ids:
        raise ValueError("No datasets found in bq_ddl_schemas within database_settings.")

    # Select default or preferred dataset (you can improve later with selection logic)
    # For now, pick the first dataset ID available in the bq_ddl_schemas dictionary
    selected_dataset = dataset_ids[0]

    # Store selected dataset for later use
    callback_context.state["selected_dataset"] = selected_dataset

    # Load schema into top-level state
    # Correctly access the schema from the 'bq_ddl_schemas' dictionary
    schema = all_ddl_schemas[selected_dataset]
    callback_context.state["bq_ddl_schema"] = schema

    # Update the agent's instruction with the selected schema
    callback_context._invocation_context.agent.instruction = (
        return_instructions_bigquery()
        + f"""

--------- The BigQuery schema for the selected dataset '{selected_dataset}'. ---------
{schema}

"""
    )


database_agent = Agent(
    model=os.getenv("BIGQUERY_AGENT_MODEL"),
    name="database_agent",
    instruction=return_instructions_bigquery(),
    tools=[
        (
            chase_db_tools.initial_bq_nl2sql
            if NL2SQL_METHOD == "CHASE"
            else tools.initial_bq_nl2sql
        ),
        tools.run_bigquery_validation,
    ],
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
