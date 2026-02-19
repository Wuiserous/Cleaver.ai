import os
import io
import asyncio
import logging
import datetime
from dotenv import set_key, load_dotenv
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from google import genai
from google.genai import types

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

import my_tools
from my_tools import get_memory_content
import browser_agent # Import the agent file

# --- CONFIGURATION ---
load_dotenv()

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
    print("❌ Error: Please set TELEGRAM_TOKEN and GEMINI_API_KEY in .env file")
    exit(1)

# Global State
GLOBAL_CHAT_ID = os.getenv("LAST_CHAT_ID")
TELEGRAM_APP = None
BOT_LOOP = None

# Add this near the top of main.py
LATEST_USER_LOCATION = {
    "latitude": None,
    "longitude": None,
    "address": "Unknown Location",
    "last_updated": None
}
# Initialize Geolocator (Give it a unique user_agent name)
GEOLOCATOR = Nominatim(user_agent="my_personal_ai_assistant_v1")

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = 'gemini-3-flash-preview'

# --- SYSTEM INSTRUCTIONS ---
SYS_INSTRUCT = """
You are an Autonomous Universal Agent. You have full control via Python.

### YOUR MEMORY (CONTEXT):
{memory_content}

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
        system_instruction=SYS_INSTRUCT.format(memory_content=get_memory_content()),
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
# --- CORE AI LOGIC ---
def process_interaction(content, from_scheduler=False):
    """
    Blocking function that talks to Gemini.
    """
    global GLOBAL_CHAT_ID

    # 1. PREPARE CONTEXT (LOCATION & TIME) - DO THIS FIRST!
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check Global Location Memory
    loc_info = "Location: Unknown (User has not shared live location yet)"
    if LATEST_USER_LOCATION["latitude"]:
        loc_info = (
            f"User Location: {LATEST_USER_LOCATION['address']}\n"
            f"GPS Coordinates: {LATEST_USER_LOCATION['latitude']}, {LATEST_USER_LOCATION['longitude']}\n"
            f"Last Updated: {LATEST_USER_LOCATION['last_updated']}"
        )

    # Create the System Header
    context_header = f"[System Context | Time: {current_time}]\n[{loc_info}]"

    # 2. CONSTRUCT PAYLOAD
    if from_scheduler:
        # If it's a scheduled job, we just send the instruction
        payload = f"{context_header}\n\n[SYSTEM JOB] Execute: {content}"
    elif isinstance(content, list):
        # Multimodal (Images + Text)
        payload = content
        payload.append(context_header)  # Append context as text
    else:
        # Standard Text Message
        payload = f"{context_header}\n\n[User Message] {content}"

    try:
        # 3. SEND TO GEMINI
        response = chat.send_message(payload)
    except Exception as e:
        error_msg = f"Gemini API Error: {e}"
        print(error_msg)
        if GLOBAL_CHAT_ID and BOT_LOOP:
            asyncio.run_coroutine_threadsafe(send_telegram_message(GLOBAL_CHAT_ID, error_msg), BOT_LOOP)
        return

    # --- AUTOMATIC FUNCTION CALLING LOOP ---
    # (This part remains exactly the same as your code)
    while response.function_calls:
        parts_to_send = []

        for call in response.function_calls:
            print(f" > 🤖 Tool Call: {call.name} | Args: {call.args}")
            tool_func = getattr(my_tools, call.name, None)
            result = "Error: Tool not found"

            if tool_func:
                try:
                    result = tool_func(**call.args)
                except Exception as e:
                    result = f"Tool Execution Exception: {e}"

            # Security Intercept Logic
            if call.name == "request_env_variable" and "SYSTEM_WAIT_ACTION" in str(result):
                if GLOBAL_CHAT_ID and TELEGRAM_APP and BOT_LOOP:
                    var_name = call.args.get('var_name')
                    msg = (f"🛑 **SECURITY INTERCEPT**\n"
                           f"The Agent needs: `{var_name}`\n"
                           f"**Reply with the key ONLY.**")
                    asyncio.run_coroutine_threadsafe(
                        send_telegram_message(GLOBAL_CHAT_ID, msg),
                        BOT_LOOP
                    )
                parts_to_send.append(types.Part.from_function_response(name=call.name, response={"result": result}))
            else:
                parts_to_send.append(types.Part.from_function_response(name=call.name, response={"result": result}))

        if parts_to_send:
            print("   < Sending Tool Results to AI...")
            try:
                response = chat.send_message(parts_to_send)
            except Exception as e:
                print(f"Error sending tool response: {e}")
                break
        else:
            break

    # 4. FINAL RESPONSE
    if response.text and TELEGRAM_APP and GLOBAL_CHAT_ID:
        if "NO_ACTION" in response.text:
            print(f"   [Heartbeat] AI decided to stay silent.")
            return

        asyncio.run_coroutine_threadsafe(
            send_telegram_message(GLOBAL_CHAT_ID, response.text),
            BOT_LOOP
        )
def get_address_from_coords(lat, lon):
    """
    Converts (Lat, Lon) -> "123 Main St, New York, USA"
    """
    try:
        location = GEOLOCATOR.reverse((lat, lon), language='en', timeout=5)
        if location:
            return location.address
    except Exception as e:
        print(f"Geocoding Error: {e}")
    return "Unknown Address"


async def handle_live_location_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Runs silently when Live Location updates.
    """
    global LATEST_USER_LOCATION

    # 1. Get Coordinates (Check both message types)
    # When you first share, it's 'message'. When you move, it's 'edited_message'.
    msg = update.edited_message if update.edited_message else update.message

    if not msg or not msg.location:
        return

    lat = msg.location.latitude
    lon = msg.location.longitude

    # 2. Update Global Lat/Lon immediately
    LATEST_USER_LOCATION["latitude"] = lat
    LATEST_USER_LOCATION["longitude"] = lon
    LATEST_USER_LOCATION["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 3. Get Address (Run in thread to avoid blocking bot)
    readable_address = await asyncio.to_thread(get_address_from_coords, lat, lon)
    LATEST_USER_LOCATION["address"] = readable_address


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


async def browser_status_handler(text, image_bytes):
    """
    Callback function passed to the Browser Agent.
    It runs on the main asyncio loop.
    """
    if GLOBAL_CHAT_ID and TELEGRAM_APP:
        try:
            # Send photo with caption
            await TELEGRAM_APP.bot.send_photo(
                chat_id=GLOBAL_CHAT_ID,
                photo=image_bytes,
                caption=f"🌐 **Browser Agent**\n`{text}`",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            print(f"Failed to send browser update: {e}")

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


# Define a startup function
async def on_startup(app):
    print("--- 🔗 Starting Browser WebSocket Server ---")
    # 1. Start the Browser Server as a background task
    app.create_task(browser_agent.start_browser_server())

    # 2. Link the Main Loop to my_tools so it can schedule tasks
    my_tools.MAIN_LOOP = asyncio.get_running_loop()

    # 3. Link the Callback so my_tools can pass it to the agent
    my_tools.GLOBAL_BROWSER_CALLBACK = browser_status_handler

    print("--- ✅ System Ready ---")


if __name__ == "__main__":
    print("--- 🚀 UNIVERSAL AI AGENT STARTED ---")

    # Link Tools Callback
    my_tools.AI_CALLBACK_FUNC = process_interaction

    # Build App with post_init
    TELEGRAM_APP = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(on_startup).build()

    # Add Handlers
    TELEGRAM_APP.add_handler(MessageHandler(
        filters.TEXT | filters.PHOTO | filters.AUDIO | filters.VOICE,
        handle_message
    ))

    # 1. Handle NORMAL messages (Text, Photo, Audio)
    TELEGRAM_APP.add_handler(MessageHandler(
        filters.TEXT | filters.PHOTO | filters.AUDIO | filters.VOICE,
        handle_message
    ))

    # 1. Live Location Updates (Handled separately so it's silent)
    # We catch BOTH normal location messages AND edited messages (updates)
    TELEGRAM_APP.add_handler(MessageHandler(
        filters.LOCATION,
        handle_live_location_update
    ))

    # 2. Standard User Messages (Text, Photo, Audio)
    # Note: We do NOT include filters.LOCATION here, so it doesn't trigger the main AI loop twice
    TELEGRAM_APP.add_handler(MessageHandler(
        filters.TEXT | filters.PHOTO | filters.AUDIO | filters.VOICE,
        handle_message
    ))

    # Run
    TELEGRAM_APP.run_polling()