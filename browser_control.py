import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# --- ROBUST IMPORTS ---
from browser_use import Agent

# Try importing Browser from the top level (Newer Versions)
try:
    from browser_use import Browser
except ImportError:
    # Fallback for older versions
    try:
        from browser_use.browser.browser import Browser
    except ImportError:
        print("❌ Critical Error: Could not import 'Browser' from browser_use.")
        exit(1)

# Try importing BrowserConfig (Newer Versions)
try:
    from browser_use import BrowserConfig
except ImportError:
    try:
        # Older versions
        from browser_use.browser.browser import BrowserConfig
    except ImportError:
        # If BrowserConfig is completely gone, we will use a dictionary or None
        print("⚠️ Warning: BrowserConfig not found. Using dictionary config fallback.")
        BrowserConfig = None

load_dotenv()

# --- CONFIGURATION ---
MODEL_NAME = "gemini-2.5-flash"
PROFILE_PATH = os.path.join(os.getcwd(), "chrome_data")

if not os.getenv("GOOGLE_API_KEY"):
    print("❌ Error: GOOGLE_API_KEY is missing from .env")
    exit(1)

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.0,
)

# --- INITIALIZE BROWSER ---
# We handle the case where BrowserConfig might be missing
try:
    if BrowserConfig:
        # Standard way
        browser = Browser(
            config=BrowserConfig(
                headless=False,  # Set to True for background mode
                disable_security=True,
            )
        )
    else:
        # Fallback: Recent versions might accept a dict or kwargs
        print("🔧 Attempting to initialize Browser with kwargs...")
        # Try passing config as a dict (works in some Pydantic implementations)
        # Or try passing arguments directly if the library changed signature
        try:
            browser = Browser(config={'headless': False, 'disable_security': True})
        except:
            # Last resort: Try no config (will run headless by default usually)
            print("⚠️ Config dictionary failed. Starting with default settings.")
            browser = Browser()

except Exception as e:
    print(f"❌ Failed to initialize Browser: {e}")
    exit(1)


async def run_browser_task(task_description: str):
    print(f"🚀 Browser Agent Starting: {task_description}")

    # Create Context (Window/Tab Settings)
    try:
        context = await browser.new_context(
            config=BrowserConfig(
                user_data_dir=PROFILE_PATH,
                browser_window_size={'width': 1280, 'height': 800}
            )
        )
    except Exception as e:
        print(f"❌ Failed to create context: {e}")
        return

    agent = Agent(
        task=task_description,
        llm=llm,
        browser_context=context,
        use_vision=True,
    )

    try:
        history = await agent.run()
        result = history.final_result()
        print(f"\n✅ RESULT:\n{result}")
        await context.close()
        return result
    except Exception as e:
        print(f"❌ Runtime Error: {e}")
        await context.close()
        return str(e)


if __name__ == "__main__":
    # Fix for Windows Asyncio Loop
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


    async def main():
        task = "Go to Google.com, search for 'DeepSeek', and tell me the first headline."
        await run_browser_task(task)


    asyncio.run(main())