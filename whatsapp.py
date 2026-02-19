import os
from google import genai
from google.genai import types
from neonize.client import NewClient
from neonize.events import ConnectedEv, MessageEv
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chat_session = gemini_client.chats.create(
    model='gemini-2.5-flash-lite',
    config=types.GenerateContentConfig(
        system_instruction='You are a Friendly Witty chatbot',
        temperature=0.5
    )
)

ALLOWED_NUMBER = "917289962452"


# --- AI LOGIC ---

def ask_gemini(prompt):
    try:
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "I'm having trouble connecting to my Gemini brain."


# --- WHATSAPP LOGIC ---

client = NewClient("whatsapp_session.sqlite3")


@client.event(ConnectedEv)
def on_connected(_: NewClient, __: ConnectedEv):
    print("\n✅ WhatsApp connected and ready!")


@client.event(MessageEv)
def on_message(cl: NewClient, v: MessageEv):
    try:
        # --- 1. Get Source Info ---
        source = v.Info.MessageSource
        chat_obj = source.Chat

        sender_phone = chat_obj.User
        sender_server = chat_obj.Server

        # --- 2. Filter Groups & Self ---
        if sender_server == "g.us" or sender_server == "broadcast":
            return
        if source.IsFromMe:
            return

        # --- 3. Extract Text ---
        message_text = ""
        if v.Message.conversation:
            message_text = v.Message.conversation
        elif v.Message.extendedTextMessage:
            message_text = v.Message.extendedTextMessage.text

        if not message_text:
            return

        # --- DEBUG PRINT ---
        print(f"DEBUG: Msg from Phone: {sender_phone} | Text: {message_text}")

        # --- 4. Logic ---
        if sender_phone == ALLOWED_NUMBER:
            print(f"\n[Incoming] {message_text}")

            try:
                # 1 represents 'Composing' status
                cl.send_chat_presence(chat_obj, 1)
            except Exception as e:
                print(f"Presence Error (Non-fatal): {e}")

            # Generate AI Response
            ai_response = ask_gemini(message_text)

            # --- SEND RESPONSE ---
            try:
                # OPTION 1: Direct Send (More reliable than reply for debugging)
                cl.send_message(chat_obj, ai_response)

                # OPTION 2: Reply (Try this if Option 1 works and you want to quote)
                # cl.reply_message(chat_obj, ai_response, v)

                print(f"[Outgoing] {ai_response}")
            except Exception as send_err:
                print(f"❌ FAILED TO SEND: {send_err}")

    except Exception as e:
        print(f"Error processing message: {e}")
        # Print full traceback to see exactly where it fails
        import traceback
        traceback.print_exc()


print("Starting WhatsApp client...")
client.connect()