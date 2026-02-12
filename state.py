import asyncio

# Shared memory across all modules
GLOBAL_STATE = {
    "stop_signal": False,
    "current_chat_id": None,
    "main_loop": None,      # The asyncio loop of the bot
    "ui_callback": None,    # The function to update the dashboard
    "browser_task": None    # <--- NEW: Holds the running browser task for cancellation
}

def check_stop_signal():
    """Helper to check if execution should stop."""
    if GLOBAL_STATE["stop_signal"]:
        raise InterruptedError("🛑 Task Terminated by User Request.")