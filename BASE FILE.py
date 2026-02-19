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


# List of tools to pass to Gemini
defined_tools = [
    execute_python_code,
    manage_skills,
    request_env_variable,
    schedule_instruction,
    perform_web_search
]


import os
import io
import asyncio
import logging
import datetime
from dotenv import set_key, load_dotenv

from google import genai
from google.genai import types

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

import my_tools

# --- CONFIGURATION ---
load_dotenv()

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
    print("❌ Error: Please set TELEGRAM_TOKEN and GEMINI_API_KEY in .env file")
    exit(1)

# Global State
GLOBAL_CHAT_ID = None
TELEGRAM_APP = None
BOT_LOOP = None

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = 'gemini-3-flash-preview'

# --- SYSTEM INSTRUCTIONS ---
SYS_INSTRUCT = """
You are an Autonomous Universal Agent. You have full control via Python.

### YOUR TOOLBOX:
1. **`execute_python_code`**: 
   - Use this to do ANYTHING. Fetch URLs, parse JSON, check stocks, update databases.
   - You have access to `requests`, `pandas`, `os`, `sys`, etc.
   - Always `print()` your final result so you can see it.

2. **`manage_skills`**:
   - Check `action='list'` first to see if you already know how to do a task.
   - If you write a script that works well, call `action='save'` to store it for the future.

3. **`request_env_variable` (SECURITY)**:
   - NEVER ask the user for keys in plain text.
   - If a script requires a key (e.g., `os.getenv('NOTION_KEY')`) and it returns None:
     - Call `request_env_variable(var_name='NOTION_KEY', reason='...')`.
     - STOP and wait. The system will handle the rest.

### OPERATING RULES:
- If asked to do a task, first check if you have a skill for it.
- If not, write a Python script. If the script fails, fix it and run again.
- Once a script works, SAVE IT.
- Output formatting: Use Markdown. Bold keys. Code blocks for data.
"""

# Create the Chat Session
# We configure tools here. The SDK automatically parses the functions.
chat = client.chats.create(
    model=MODEL_ID,
    config=types.GenerateContentConfig(
        system_instruction=SYS_INSTRUCT,
        tools=my_tools.defined_tools,  # Pass the list of function objects
        temperature=0.0
    )
)


# --- TELEGRAM SENDER ---
async def send_telegram_message(chat_id, text):
    """Async helper to send messages back to Telegram."""
    if not text: return
    try:
        await TELEGRAM_APP.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        # Fallback to plain text if Markdown fails
        await TELEGRAM_APP.bot.send_message(chat_id=chat_id, text=text)


# --- CORE AI LOGIC ---
def process_interaction(content, from_scheduler=False):
    """
    Blocking function that talks to Gemini.
    It must be run in a separate thread (via asyncio.to_thread) to avoid freezing Telegram.
    """
    global GLOBAL_CHAT_ID

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Construct the payload
    if from_scheduler:
        payload = f"[SYSTEM JOB | {current_time}] Execute: {content}"
    elif isinstance(content, list):
        # Multimodal content (text + images)
        payload = content
        # Add context string as text part
        payload.append(f"\n[User Context | {current_time}]")
    else:
        payload = f"[User Message | {current_time}] {content}"

    try:
        # Send message to Gemini
        response = chat.send_message(payload)
    except Exception as e:
        error_msg = f"Gemini API Error: {e}"
        print(error_msg)
        if GLOBAL_CHAT_ID and BOT_LOOP:
            asyncio.run_coroutine_threadsafe(send_telegram_message(GLOBAL_CHAT_ID, error_msg), BOT_LOOP)
        return

    # --- AUTOMATIC FUNCTION CALLING LOOP ---
    # The SDK wraps the response. We check if the model wants to call tools.
    while response.function_calls:
        parts_to_send = []

        for call in response.function_calls:
            print(f" > 🤖 Tool Call: {call.name} | Args: {call.args}")

            # 1. Execute the tool
            tool_func = getattr(my_tools, call.name, None)

            result = "Error: Tool not found"
            if tool_func:
                try:
                    result = tool_func(**call.args)
                except Exception as e:
                    result = f"Tool Execution Exception: {e}"

            # 2. Check for Security Intercept
            # If the tool paused execution to ask for a password, we notify user and break loop
            if call.name == "request_env_variable" and "SYSTEM_WAIT_ACTION" in str(result):
                if GLOBAL_CHAT_ID and TELEGRAM_APP and BOT_LOOP:
                    var_name = call.args.get('var_name')
                    msg = (f"🛑 **SECURITY INTERCEPT**\n"
                           f"The Agent needs: `{var_name}`\n\n"
                           f"**Reply with the key ONLY.**\n"
                           f"I will save it securely, delete your message, and resume the Agent.")
                    asyncio.run_coroutine_threadsafe(
                        send_telegram_message(GLOBAL_CHAT_ID, msg),
                        BOT_LOOP
                    )

                # We still send the result to Gemini so it knows it's waiting
                parts_to_send.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"result": result}
                    )
                )
                # We do NOT break here immediately, we let Gemini receive the 'Wait' status.
                # Usually Gemini will stop generating after receiving this status if instructed well.

            else:
                # Standard Tool Response
                parts_to_send.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"result": result}
                    )
                )

        # Send tool outputs back to Gemini
        if parts_to_send:
            print("   < Sending Tool Results to AI...")
            try:
                response = chat.send_message(parts_to_send)
            except Exception as e:
                print(f"Error sending tool response: {e}")
                break
        else:
            break

    # Final Response to User (Text)
    if response.text and TELEGRAM_APP and GLOBAL_CHAT_ID:
        asyncio.run_coroutine_threadsafe(
            send_telegram_message(GLOBAL_CHAT_ID, response.text),
            BOT_LOOP
        )


# --- SECURITY INTERCEPTOR ---
async def handle_security_interceptor(update: Update):
    """
    Checks if we are waiting for a password. If so, saves it and stops AI processing.
    """
    if my_tools.PENDING_ENV_VAR["waiting"]:
        var_name = my_tools.PENDING_ENV_VAR["var_name"]
        secret_value = update.message.text

        print(f"🔒 Intercepted secret for {var_name}")

        # 1. Save to .env file (Persistent)
        set_key(".env", var_name, secret_value)

        # 2. Load into immediate environment
        os.environ[var_name] = secret_value

        # 3. Reset Flag
        my_tools.PENDING_ENV_VAR["waiting"] = False
        my_tools.PENDING_ENV_VAR["var_name"] = None

        # 4. Delete user message (Security)
        try:
            await update.message.delete()
        except:
            await update.message.reply_text("⚠️ I couldn't delete your message. Please delete it manually.")

        await update.message.reply_text(f"✅ **{var_name}** saved securely. Resuming Agent...")

        # 5. Resume AI by manually triggering the next turn
        await asyncio.to_thread(
            process_interaction,
            f"[SYSTEM NOTICE] User has provided {var_name}. It is now available in os.environ. Proceed with the task."
        )
        return True  # Handled
    return False  # Not handled


# --- MESSAGE HANDLER ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global GLOBAL_CHAT_ID, BOT_LOOP

    if not update.effective_chat: return

    user_id = update.effective_chat.id
    GLOBAL_CHAT_ID = user_id
    if BOT_LOOP is None: BOT_LOOP = asyncio.get_running_loop()

    # 1. CHECK SECURITY INTERCEPTOR
    if await handle_security_interceptor(update):
        return

    # 2. PROCESS CONTENT (Text/Photo/Audio)
    gemini_content = []

    # Text
    text_input = update.message.text or update.message.caption
    if text_input:
        gemini_content.append(text_input)

    # Photos
    if update.message.photo:
        photo_file = await update.message.photo[-1].get_file()
        f_stream = io.BytesIO()
        await photo_file.download_to_memory(out=f_stream)
        # Using PIL or Bytes for Gemini
        gemini_content.append(types.Part.from_bytes(data=f_stream.getvalue(), mime_type="image/jpeg"))

    # Audio/Voice
    if update.message.voice or update.message.audio:
        media_obj = update.message.voice or update.message.audio
        f_file = await media_obj.get_file()
        f_stream = io.BytesIO()
        await f_file.download_to_memory(out=f_stream)
        mime = "audio/ogg" if update.message.voice else media_obj.mime_type
        gemini_content.append(types.Part.from_bytes(data=f_stream.getvalue(), mime_type=mime))

    if gemini_content:
        # Indicate typing status
        await context.bot.send_chat_action(chat_id=user_id, action="typing")

        # If just one text item, pass string. If mixed, pass list.
        content_to_pass = gemini_content[0] if len(gemini_content) == 1 and isinstance(gemini_content[0],
                                                                                       str) else gemini_content

        # RUN BLOCKING AI TASK IN THREAD
        await asyncio.to_thread(process_interaction, content_to_pass)


if __name__ == "__main__":
    print("--- 🚀 UNIVERSAL AI AGENT STARTED ---")

    # 1. Link the Tools Callback to the Main Logic
    # This allows the APScheduler in my_tools to trigger process_interaction here
    my_tools.AI_CALLBACK_FUNC = process_interaction

    # 2. Build Telegram Bot
    TELEGRAM_APP = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # 3. Add Handlers
    TELEGRAM_APP.add_handler(MessageHandler(
        filters.TEXT | filters.PHOTO | filters.AUDIO | filters.VOICE,
        handle_message
    ))

    # 4. Run
    TELEGRAM_APP.run_polling()



#++++greatest achievement++++