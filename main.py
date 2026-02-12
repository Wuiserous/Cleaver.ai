import os
import io
import asyncio
import logging
import datetime
import time
import socket
from dotenv import set_key, load_dotenv

from google import genai
from google.genai import types

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from telegram.error import NetworkError, TimedOut

import my_tools
from memory import memory
import browser_interface

# --- CONFIGURATION ---
load_dotenv()
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

GLOBAL_CHAT_ID = None
TELEGRAM_APP = None
BOT_LOOP = None

# UI State Tracking
CURRENT_STATUS_MSG_ID = None
LAST_STATUS_TEXT = ""

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = 'gemini-3-flash-preview'

SYS_INSTRUCT = """
You are an Autonomous Universal Agent with Persistent Memory.
Adapt to user preference, if you find any recurring events set by user, suggest setting a reminder
Use all the tools efficiently and effectively.

### MEMORY & TOOLS:
1. **Memory**: You have access to past conversations. Do not repeat questions you already know the answer to.
2. **Skill Library**: You do not have every tool built-in. 
   - ALWAYS check `manage_skills(action='search', search_query='...')` before writing new code.
   - If a skill exists, get the code and run it using `execute_python_code`.
   - If you write a NEW useful script, save it using `manage_skills(action='save', ...)`.
3. **Browser**: Use `use_browser` for any web interaction (Amazon, Shopping, Research, youtube).
### OUTPUT:
- Be concise. Use Markdown.
"""

chat = client.chats.create(
    model=MODEL_ID,
    config=types.GenerateContentConfig(
        system_instruction=SYS_INSTRUCT,
        tools=my_tools.defined_tools,
        temperature=0.0
    )
)


# --- UI HELPER FUNCTIONS ---

async def update_status_message(text: str):
    """
    Updates the 'Loading...' message on Telegram.
    This gives the user visual feedback on what the agent is doing.
    """
    global CURRENT_STATUS_MSG_ID, GLOBAL_CHAT_ID, LAST_STATUS_TEXT
    if not GLOBAL_CHAT_ID or not TELEGRAM_APP: return
    if text == LAST_STATUS_TEXT: return  # Debounce

    LAST_STATUS_TEXT = text
    formatted_text = f"⏳ **{text}**"

    try:
        if CURRENT_STATUS_MSG_ID:
            await TELEGRAM_APP.bot.edit_message_text(
                chat_id=GLOBAL_CHAT_ID,
                message_id=CURRENT_STATUS_MSG_ID,
                text=formatted_text,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            msg = await TELEGRAM_APP.bot.send_message(
                chat_id=GLOBAL_CHAT_ID,
                text=formatted_text,
                parse_mode=ParseMode.MARKDOWN
            )
            CURRENT_STATUS_MSG_ID = msg.message_id
    except Exception as e:
        # If message not found (user deleted it), send a new one
        try:
            msg = await TELEGRAM_APP.bot.send_message(
                chat_id=GLOBAL_CHAT_ID,
                text=formatted_text,
                parse_mode=ParseMode.MARKDOWN
            )
            CURRENT_STATUS_MSG_ID = msg.message_id
        except:
            pass


async def clear_status_message():
    """Deletes the status message when the agent finishes."""
    global CURRENT_STATUS_MSG_ID, GLOBAL_CHAT_ID
    if CURRENT_STATUS_MSG_ID and GLOBAL_CHAT_ID:
        try:
            await TELEGRAM_APP.bot.delete_message(chat_id=GLOBAL_CHAT_ID, message_id=CURRENT_STATUS_MSG_ID)
        except:
            pass
        CURRENT_STATUS_MSG_ID = None


async def send_telegram_message(chat_id, text):
    try:
        await TELEGRAM_APP.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN)
    except:
        await TELEGRAM_APP.bot.send_message(chat_id=chat_id, text=text)


# --- LOGIC CORE ---

def process_interaction(content, from_scheduler=False):
    """Main Agent Logic: Memory -> Gemini -> Tools -> Response"""
    global GLOBAL_CHAT_ID, BOT_LOOP

    # Debug print to see if we have a chat ID during the job
    print(f"DEBUG: Processing interaction. Chat ID is: {GLOBAL_CHAT_ID}")

    if not GLOBAL_CHAT_ID:
        print("⚠️ Cannot send response: GLOBAL_CHAT_ID is None. Waiting for a user message first.")
        return
    # 1. Start UI
    asyncio.run_coroutine_threadsafe(update_status_message("Thinking..."), BOT_LOOP)

    user_text = content
    if isinstance(content, list):
        user_text = next((x for x in content if isinstance(x, str)), "Image Input")

    relevant_context = memory.retrieve_context(user_text)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    context_block = f"\n\n[💾 MEMORY RECALL]\n{relevant_context}\n[END MEMORY]" if relevant_context else ""

    if from_scheduler:
        full_prompt = f"[SYSTEM JOB | {current_time}] {content} {context_block}"
    elif isinstance(content, list):
        full_prompt = content
        full_prompt.append(f"\n[User Context | {current_time}] {context_block}")
    else:
        full_prompt = f"[User Message | {current_time}] {content} {context_block}"

    try:
        response = chat.send_message(full_prompt)
        if isinstance(user_text, str):
            memory.save_interaction("user", user_text)

    except Exception as e:
        asyncio.run_coroutine_threadsafe(clear_status_message(), BOT_LOOP)
        print(f"API Error: {e}")
        return

    # 2. Tool Loop
    while response.function_calls:
        parts_to_send = []
        for call in response.function_calls:
            print(f" > 🤖 Tool: {call.name}")

            # Update UI
            asyncio.run_coroutine_threadsafe(update_status_message(f"Running Tool: {call.name}"), BOT_LOOP)

            tool_func = getattr(my_tools, call.name, None)
            result = "Error: Tool not found"

            if tool_func:
                try:
                    result = tool_func(**call.args)
                except Exception as e:
                    result = f"Exception: {e}"

            # Intercept Key Requests
            if call.name == "request_env_variable" and "SYSTEM_WAIT_ACTION" in str(result):
                if GLOBAL_CHAT_ID:
                    var_name = call.args.get('var_name')
                    asyncio.run_coroutine_threadsafe(clear_status_message(), BOT_LOOP)
                    asyncio.run_coroutine_threadsafe(
                        send_telegram_message(GLOBAL_CHAT_ID, f"🛑 **Need Key**: `{var_name}`\nReply with key ONLY."),
                        BOT_LOOP
                    )
                # We return here to stop the loop, waiting for user input (handled in handle_message)

            parts_to_send.append(types.Part.from_function_response(name=call.name, response={"result": result}))

        if parts_to_send:
            asyncio.run_coroutine_threadsafe(update_status_message("Analyzing Tool Output..."), BOT_LOOP)
            response = chat.send_message(parts_to_send)
        else:
            break

    # 3. Cleanup & Response
    asyncio.run_coroutine_threadsafe(clear_status_message(), BOT_LOOP)

    if response.text:
        memory.save_interaction("model", response.text)
        if GLOBAL_CHAT_ID:
            asyncio.run_coroutine_threadsafe(
                send_telegram_message(GLOBAL_CHAT_ID, response.text),
                BOT_LOOP
            )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global GLOBAL_CHAT_ID, BOT_LOOP
    if not update.effective_chat: return
    # Check if we need to save/update the Chat ID
    if GLOBAL_CHAT_ID != update.effective_chat.id:
        GLOBAL_CHAT_ID = update.effective_chat.id
        # Save to file
        set_key(".env", "LAST_CHAT_ID", str(GLOBAL_CHAT_ID))
        # Update current process memory immediately
        os.environ["LAST_CHAT_ID"] = str(GLOBAL_CHAT_ID)
        print(f"💾 Saved Chat ID {GLOBAL_CHAT_ID} to .env")

    if BOT_LOOP is None: BOT_LOOP = asyncio.get_running_loop()

    # Security Interceptor (For setting API Keys mid-run)
    if my_tools.PENDING_ENV_VAR["waiting"]:
        var_name = my_tools.PENDING_ENV_VAR["var_name"]
        secret = update.message.text.strip()
        set_key(".env", var_name, secret)
        os.environ[var_name] = secret
        load_dotenv(override=True)
        my_tools.PENDING_ENV_VAR["waiting"] = False
        await update.message.reply_text(f"✅ **{var_name}** saved. Resuming...")
        await asyncio.to_thread(process_interaction, f"[SYSTEM] Key {var_name} provided. Resume task.")
        return

    # Gather Content
    gemini_content = []
    text = update.message.text or update.message.caption
    if text: gemini_content.append(text)

    if update.message.photo:
        f = await update.message.photo[-1].get_file()
        b = io.BytesIO()
        await f.download_to_memory(b)
        gemini_content.append(types.Part.from_bytes(data=b.getvalue(), mime_type="image/jpeg"))

    if update.message.voice or update.message.audio:
        media = update.message.voice or update.message.audio
        f = await media.get_file()
        b = io.BytesIO()
        await f.download_to_memory(b)
        mime = "audio/ogg" if update.message.voice else media.mime_type
        gemini_content.append(types.Part.from_bytes(data=b.getvalue(), mime_type=mime))

    if gemini_content:
        await context.bot.send_chat_action(chat_id=GLOBAL_CHAT_ID, action="typing")
        payload = gemini_content[0] if len(gemini_content) == 1 and isinstance(gemini_content[0],
                                                                               str) else gemini_content
        await asyncio.to_thread(process_interaction, payload)


# --- ROBUST STARTUP & CONNECTION MANAGER ---

def wait_for_internet():
    """Blocking function that waits for internet connection."""
    while True:
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return
        except OSError:
            print("⚠️ No Internet. Retrying in 5s...")
            time.sleep(5)


async def main_startup():
    global GLOBAL_CHAT_ID, BOT_LOOP
    print("--- 🚀 ROBUST VECTOR AI AGENT STARTED ---")
    # Force reload .env to catch changes made in previous runs
    load_dotenv(override=True)

    saved_chat_id = os.getenv("LAST_CHAT_ID")
    if saved_chat_id:
        GLOBAL_CHAT_ID = int(saved_chat_id)
        print(f"✅ Recovered Chat ID: {GLOBAL_CHAT_ID}")
    else:
        print("ℹ️ No Chat ID found in .env. Bot will wait for first user message.")

    wait_for_internet()

    # 1. Capture the running loop immediately
    loop = asyncio.get_running_loop()
    BOT_LOOP = loop  # <--- FIX: Initialize it here
    my_tools.MAIN_LOOP = loop  # <--- Ensure tools also have it)
    my_tools.AI_CALLBACK_FUNC = process_interaction
    my_tools.STATUS_CALLBACK_FUNC = update_status_message

    # 2. Start Browser Bridge
    await browser_interface.start_browser_server()

    # 3. Setup Bot
    global TELEGRAM_APP
    TELEGRAM_APP = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    TELEGRAM_APP.add_handler(MessageHandler(filters.ALL, handle_message))
    await TELEGRAM_APP.initialize()
    await TELEGRAM_APP.start()

    print("🤖 Telegram Bot Polling...")

    # 4. Infinite Polling with Crash Recovery
    while True:
        try:
            # start_polling is usually non-blocking if using the updater directly,
            # but here we want to handle the loop ourselves for finer control.
            await TELEGRAM_APP.updater.start_polling()

            # Keep the main coroutine alive
            stop_signal = asyncio.Event()
            await stop_signal.wait()

        except (NetworkError, TimedOut, ConnectionError) as e:
            print(f"❌ Network Drop: {e}. Sleeping 10s...")
            try:
                await TELEGRAM_APP.updater.stop()
            except:
                pass
            await asyncio.sleep(10)
            wait_for_internet()
            print("♻️ Reconnecting...")

        except Exception as e:
            print(f"❌ Critical Error: {e}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main_startup())
    except KeyboardInterrupt:
        print("Shutting down...")