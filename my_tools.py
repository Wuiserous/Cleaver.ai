import os
import sys
import io
import json
import traceback
import subprocess
import contextlib
import datetime
import asyncio
import logging
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.date import DateTrigger
from google import genai
from google.genai import types
import browser_interface
from memory import memory

load_dotenv()

# --- GLOBAL CONFIG ---
PENDING_ENV_VAR = {"waiting": False, "var_name": None}

# Global hooks for Main.py to inject functionality
AI_CALLBACK_FUNC = None  # The function to run the AI logic
STATUS_CALLBACK_FUNC = None  # The function to update the Telegram UI
MAIN_LOOP = None  # The main asyncio loop

# --- PERSISTENT SCHEDULER SETUP ---
# We use SQLite to save jobs. If the script restarts, jobs are reloaded.
jobstores = {
    'default': SQLAlchemyJobStore(url='sqlite:///jobs.sqlite')
}

# NEW: Job defaults to prevent "Missed" warnings
job_defaults = {
    'misfire_grace_time': 60, # Allow jobs to run up to 60 seconds late
    'coalesce': True          # If multiple instances of the same job missed, run once
}

scheduler = BackgroundScheduler(jobstores=jobstores, job_defaults=job_defaults)
scheduler.start()


# --- STATUS HELPER ---
def update_status_sync(text: str):
    """Safe wrapper to call the async status update from synchronous tools."""
    if STATUS_CALLBACK_FUNC and MAIN_LOOP:
        if asyncio.iscoroutinefunction(STATUS_CALLBACK_FUNC):
            asyncio.run_coroutine_threadsafe(STATUS_CALLBACK_FUNC(text), MAIN_LOOP)


# --- JOB RUNNER (Must be top-level for Pickle/SQLite) ---
def _scheduled_job_runner(instruction: str):
    """Wrapper that runs when the timer goes off."""
    print(f"⏰ Job Triggered: {instruction}")
    if AI_CALLBACK_FUNC:
        # We pass 'from_scheduler=True' so the bot knows it's a self-initiated task
        AI_CALLBACK_FUNC(instruction, from_scheduler=True)
    else:
        print("❌ Error: AI Core not ready to handle job.")


# --- TOOLS ---

def execute_python_code(code: str, install_dependencies: bool = False) -> str:
    """
    Executes Python code. Can install libraries.
    """
    update_status_sync("🐍 Executing Python Code...")
    print(f"\n[🐍 EXEC] Running Code...")

    if install_dependencies:
        try:
            import re
            imports = re.findall(r'^(?:import|from) (\w+)', code, re.MULTILINE)
            for lib in imports:
                if lib not in sys.modules:
                    update_status_sync(f"📦 Installing {lib}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        except Exception as e:
            return f"Dep Install Error: {e}"

    output_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
            exec_globals = {
                'os': os, 'sys': sys, 'json': json, 'datetime': datetime,
                'print': print, 'memory': memory
            }
            exec(code, exec_globals)
        result = output_buffer.getvalue()
        return f"Result:\n{result}" if result else "(No Output)"
    except Exception:
        return f"Runtime Error:\n{traceback.format_exc()}"


def manage_skills(action: str, name: str = None, code: str = None, description: str = None,
                  search_query: str = None) -> str:
    """Manages the Vector DB Skill Library."""
    if action == "save":
        if not name or not code or not description:
            return "Error: 'name', 'code', and 'description' required to save."
        return memory.save_skill(name, code, description, usage_example=description)

    elif action == "search":
        if not search_query: return "Error: 'search_query' required."
        result = memory.retrieve_skill(search_query)
        if result:
            return f"Found Skill: {result['name']}\nDescription: {result['description']}\nCODE:\n{result['code']}"
        return "No relevant skill found."

    elif action == "list":
        skills = memory.list_all_skills()
        return f"Current Skills in DB: {skills}"

    return "Invalid action."


def request_env_variable(var_name: str, reason: str) -> str:
    """Request a missing API key from the user."""
    if os.getenv(var_name):
        return f"SUCCESS: {var_name} exists."

    PENDING_ENV_VAR["waiting"] = True
    PENDING_ENV_VAR["var_name"] = var_name
    return f"[SYSTEM_WAIT_ACTION] Paused. Asking user for {var_name}."


def perform_web_search(query: str) -> str:
    """Google Search via Gemini Grounding."""
    update_status_sync(f"🌍 Searching Google: {query}")
    print(f"[SEARCH] {query}")
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            )
        )
        # Check if grounding metadata exists
        if response.candidates and response.candidates[0].grounding_metadata:
            return response.text
        return response.text
    except Exception as e:
        return f"Search Error: {e}"


def schedule_instruction(instruction: str, time_iso: str = None) -> str:
    """
    Schedule a task. ISO format: YYYY-MM-DDTHH:MM:SS
    Example: 2025-10-27T15:30:00
    """
    try:
        run_date = datetime.datetime.fromisoformat(time_iso)
        # We accept a job ID to make it unique, but APScheduler handles it.
        # Note: we pass the FUNCTION reference, not a lambda, so it can be pickled.
        scheduler.add_job(_scheduled_job_runner, 'date', run_date=run_date, args=[instruction], misfire_grace_time=60)
        return f"✅ Scheduled persistent job for {time_iso}"
    except Exception as e:
        return f"Error: {e}"


def use_browser(task_description: str) -> str:
    """
    Controls the computer's browser to perform actions.
    """
    if not browser_interface.relay.ws:
        return "❌ Error: Browser Extension not connected. Please open the browser."

    update_status_sync(f"🌐 Starting Browser Agent...")

    if MAIN_LOOP:
        try:
            # We pass the status callback to the browser agent
            future = asyncio.run_coroutine_threadsafe(
                browser_interface.execute_browser_task(task_description, status_callback=STATUS_CALLBACK_FUNC),
                MAIN_LOOP
            )
            return future.result()
        except Exception as e:
            return f"Thread Bridge Error: {e}"
    else:
        return "Error: MAIN_LOOP not set in my_tools."


# Tool Definitions for Gemini
defined_tools = [
    execute_python_code,
    manage_skills,
    request_env_variable,
    perform_web_search,
    schedule_instruction,
    use_browser
]