import os
import sys
import io
import json
import traceback
import subprocess
import contextlib
import datetime
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from google import genai
from google.genai import types
import browser_agent
import asyncio
# Load existing environment variables
load_dotenv()

# --- 1. GLOBAL STATE & CONFIG ---
SKILLS_FILE = "skills_library.json"
PENDING_ENV_VAR = {
    "waiting": False,
    "var_name": None
}

# Initialize Scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Callback placeholder (assigned in main.py)
AI_CALLBACK_FUNC = None

# Ensure skills file exists
if not os.path.exists(SKILLS_FILE):
    with open(SKILLS_FILE, 'w') as f:
        json.dump({}, f)


# --- 2. HELPER FUNCTIONS ---

def _load_skills():
    try:
        with open(SKILLS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


# --- ADD THIS GLOBAL STATE ---
# We need a reference to the Main Async Loop (Telegram's loop)
MAIN_LOOP = None
BROWSER_TASK = None

# 1. Add this near the top with other imports
MEMORY_FILE = "memory.md"

# Ensure memory file exists
if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, 'w') as f:
        f.write("# User Context & Long Term Memory\n- User Name: Aman\n")


# 2. Add this Helper Function (Used by main.py to read memory)
def get_memory_content():
    """Reads the memory file to inject into System Instructions."""
    try:
        with open(MEMORY_FILE, 'r') as f:
            return f.read()
    except:
        return "No memory yet."


# 3. Add this TOOL Function (Used by the Agent to save facts)
def update_memory(category: str, fact: str) -> str:
    """
    Saves a permanent fact about the user or world.
    Args:
        category: Section header (e.g., 'Preferences', 'Work', 'Personal').
        fact: The information to save.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    entry = f"\n- **[{category}]** {fact} (Added: {timestamp})"

    with open(MEMORY_FILE, "a") as f:
        f.write(entry)

    return f"Memory updated: [{category}] {fact}"


# --- ADD THESE NEW TOOLS ---

def start_browser_task(task_description: str) -> str:
    """
    Starts the autonomous browser agent in the background.
    Args:
        task_description: What the browser should do (e.g., "Go to google and search for puppies").
    """
    global BROWSER_TASK

    if not MAIN_LOOP:
        return "Error: Main Event Loop not linked. Cannot start async browser."

    if browser_agent.browser_state.is_running:
        return "Error: Browser is already running a task. Use 'inject_browser_instruction' to update it or 'stop_browser_task' first."

    # Define the callback wrapper to use the global SEND_PHOTO_FUNC (we will set this in main.py)
    async def _runner():
        # We assume AI_CALLBACK_FUNC is available (from your original code) or we pass a specific sender
        # This will be triggered inside the main loop
        await browser_agent.run_agent(task_description, status_callback=GLOBAL_BROWSER_CALLBACK)

    # Schedule the task on the main thread
    BROWSER_TASK = asyncio.run_coroutine_threadsafe(_runner(), MAIN_LOOP)

    return "Browser Agent started in the background. You will receive screenshots via Telegram."


def stop_browser_task() -> str:
    """Stops the current browser automation task immediately."""
    if not browser_agent.browser_state.is_running:
        return "Browser is not currently running."

    browser_agent.browser_state.stop_requested = True
    return "Stop signal sent to Browser Agent."


def inject_browser_instruction(instruction: str) -> str:
    """
    Sends new instructions to the running browser agent without stopping it.
    Useful if the agent is stuck or needs clarification.
    """
    if not browser_agent.browser_state.is_running:
        return "Browser is not running. Use 'start_browser_task' instead."

    browser_agent.browser_state.user_feedback = instruction
    return f"Instruction injected: {instruction}"

def _save_skill_to_disk(name, code, description):
    skills = _load_skills()
    skills[name] = {
        "code": code,
        "description": description,
        "updated_at": datetime.datetime.now().isoformat()
    }
    with open(SKILLS_FILE, 'w') as f:
        json.dump(skills, f, indent=4)


def trigger_ai_action(instruction: str):
    """Callback used by the scheduler to wake up the AI."""
    print(f"\n[⏰ SCHEDULER FIRED]: {instruction}")
    if AI_CALLBACK_FUNC:
        # Pass instruction to the main loop
        AI_CALLBACK_FUNC(instruction, from_scheduler=True)
    else:
        print("Error: AI_CALLBACK_FUNC is not linked.")


# --- 3. THE TOOLS ---

def request_env_variable(var_name: str, reason: str) -> str:
    """
    CRITICAL SECURITY TOOL.
    Call this when you need an API Key, Password, or Token that is missing (returns None in os.getenv).
    It will PAUSE execution and ask the user to provide it securely via Telegram.

    Args:
        var_name: The environment variable name (e.g., 'NOTION_API_KEY').
        reason: Why it is needed.
    """
    # 1. Check if we already have it in memory
    if os.getenv(var_name):
        return f"SUCCESS: {var_name} is already loaded and ready to use in os.getenv()."

    # 2. Trigger the Interceptor in main.py
    PENDING_ENV_VAR["waiting"] = True
    PENDING_ENV_VAR["var_name"] = var_name

    # Return a status message to the model so it knows to wait
    return f"[SYSTEM_WAIT_ACTION] Execution paused. Requesting {var_name} from user. Do not generate further code until confirmed."


def manage_skills(action: str, skill_name: str = None, code: str = None, description: str = None) -> str:
    """
    Manages the AI's long-term memory of Python scripts (Skills).

    Args:
        action: 'list' (show all skills), 'save' (store a working script), 'get' (retrieve code).
        skill_name: The unique ID of the skill.
        code: The Python code (only for 'save').
        description: Short summary (only for 'save').
    """
    if action == "list":
        skills = _load_skills()
        return f"Available Skills: {list(skills.keys())}"

    if action == "save":
        if not skill_name or not code:
            return "Error: Name and Code required for saving."
        _save_skill_to_disk(skill_name, code, description)
        return f"Skill '{skill_name}' saved to library."

    if action == "get":
        skills = _load_skills()
        if skill_name in skills:
            return f"Code for {skill_name}:\n{skills[skill_name]['code']}"
        return "Skill not found."

    return "Invalid action. Use 'list', 'save', or 'get'."


def execute_python_code(code: str, install_dependencies: bool = False) -> str:
    """
    THE UNIVERSAL TOOL. Executes Python code to perform ANY task.
    Can access the internet, file system, and API keys via os.getenv().

    Args:
        code: Valid Python script.
        install_dependencies: If True, tries to pip install missing modules found in imports.
    """
    print(f"\n[🐍 PYTHON EXECUTOR] Running...\n")

    # 1. Auto-Dependency Installation
    if install_dependencies:
        try:
            import re
            imports = re.findall(r'^(?:import|from) (\w+)', code, re.MULTILINE)
            for lib in imports:
                if lib not in sys.modules:
                    print(f"Installing missing lib: {lib}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        except Exception as e:
            return f"Dependency Install Error: {e}"

    # 2. Execution Sandbox
    output_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
            exec_globals = {
                'os': os,
                'sys': sys,
                'json': json,
                'datetime': datetime,
                'print': print,
                # Add commonly used libraries here if they are imported inside the exec
            }
            exec(code, exec_globals)

        result = output_buffer.getvalue()
        if not result:
            result = "(Code ran successfully but produced no output. Did you forget to print?)"
        return f"Execution Result:\n{result}"

    except Exception:
        return f"Runtime Error:\n{traceback.format_exc()}"


def perform_web_search(query: str) -> str:
    """
    Performs a Google Search using Gemini's grounding tool to get real-time info.

    Args:
        query: The search query (e.g. "Who won the Super Bowl 2024?").
    """
    print(f"[SEARCH] 🔍 {query}")

    # We create a temporary client just for this tool action to keep it isolated,
    # or you can reuse the main client if passed down.
    # Using environment variable for key is best practice.
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not set."

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            )
        )

        # Grounding metadata contains the search results, but the text contains the synthesized answer.
        # We return the text so the main AI can use it.
        return response.text
    except Exception as e:
        return f"Search Error: {e}"


def schedule_instruction(instruction: str, time_iso: str = None, cron_expression: str = None) -> str:
    """
    Schedules a task for the future.

    Args:
        instruction: What to do (e.g. "Check tesla stock").
        time_iso: ISO format date "YYYY-MM-DDTHH:MM:SS" for one-time execution.
        cron_expression: Cron string "*/5 * * * *" for recurring execution.
    """
    job_id = f"job_{int(datetime.datetime.now().timestamp())}"
    try:
        if time_iso:
            run_date = datetime.datetime.fromisoformat(time_iso)
            scheduler.add_job(trigger_ai_action, DateTrigger(run_date=run_date), args=[instruction], id=job_id)
            return f"Scheduled for {time_iso}"
        elif cron_expression:
            scheduler.add_job(trigger_ai_action, CronTrigger.from_crontab(cron_expression), args=[instruction],
                              id=job_id)
            return f"Scheduled recurring (Cron: {cron_expression})"
        return "Error: Provide time_iso or cron_expression."
    except Exception as e:
        return f"Schedule Error: {e}"

# In my_tools.py

def run_proactive_check():
    """
    Called by scheduler. Triggers the AI to reflect on context.
    """
    if AI_CALLBACK_FUNC:
        # Get current hour to give context-aware prompts
        current_hour = datetime.datetime.now().hour

        # Dynamic Prompting based on time of day
        if 8 <= current_hour < 9:
            # Morning Briefing Mode (Force a summary)
            prompt = (
                "[SYSTEM EVENT: MORNING BRIEFING] "
                "It is morning. Please execute the following:"
                "1. Check the user's active goals."
                "2. Perform a web search for 'top tech news today' or user interests."
                "3. Compile a short morning briefing message."
                "Do NOT reply with NO_ACTION."
            )
        else:
            # Standard Heartbeat
            prompt = (
                "[SYSTEM HEARTBEAT] Proactive State Check."
                "1. Check 'active_goals.json' using manage_goals(action='list')."
                "2. If there are pending goals, remind the user or suggest the next step."
                "3. If nothing is pending, you may reply with 'NO_ACTION' to stay silent."
                "Only speak if you have value to add."
            )

        AI_CALLBACK_FUNC(prompt, from_scheduler=True)

scheduler.add_job(run_proactive_check, 'interval', minutes=25, id='proactive_heartbeat')


def manage_goals(action: str, goal: str = None, status: str = "pending") -> str:
    """
    Manage background goals.
    Args:
        action: 'add', 'list', 'complete', 'remove'
        goal: The goal description.
    """
    file_path = "active_goals.json"
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f: json.dump([], f)

    with open(file_path, 'r') as f:
        goals = json.load(f)

    if action == 'add':
        goals.append({"goal": goal, "status": "pending", "added_at": str(datetime.datetime.now())})
        msg = f"Goal added: {goal}"
    elif action == 'list':
        # Filter for pending goals
        pending = [g for g in goals if g['status'] == 'pending']
        return f"Current Active Goals:\n{json.dumps(pending, indent=2)}"
    elif action == 'complete':
        # logic to mark as done
        for g in goals:
            if goal in g['goal']: g['status'] = 'completed'
        msg = f"Marked as complete: {goal}"

    with open(file_path, 'w') as f:
        json.dump(goals, f)
    return msg


# In my_tools.py

def get_my_live_location() -> str:
    """
    Retrieves the user's current GPS coordinates from the background Live Location tracker.
    Returns: Latitude, Longitude, and Last Update Time.
    """
    # We need to access the variable from main.py.
    # Since tools are separate, usually we pass this data via the system prompt,
    # but for a quick tool access, we can try to import it or use a shared config.

    # Simplest way for your structure:
    # The AI doesn't need to 'call' this tool if we inject it into the Context.
    # See Step 5 below.
    return "Location is injected into System Context."

# List of tools to pass to Gemini
defined_tools = [
    execute_python_code,
    manage_skills,
    request_env_variable,
    schedule_instruction,
    perform_web_search,
    start_browser_task,
    stop_browser_task,
    inject_browser_instruction,
    update_memory,
    manage_goals
]

GLOBAL_BROWSER_CALLBACK = None