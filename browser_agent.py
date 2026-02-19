import asyncio
import json
import base64
import os
import time
from aiohttp import web, WSMsgType
from google import genai
from google.genai import types

# --- CONFIGURATION ---
PORT = 18792
MODEL_ID = "gemini-3-flash-preview"
TIMEOUT_CDP = 30.0


# --- SHARED STATE CONTROLLER ---
class BrowserController:
    def __init__(self):
        self.is_running = False
        self.stop_requested = False
        self.user_feedback = None
        self.latest_url = ""


browser_state = BrowserController()

SYSTEM_INSTRUCTION = """
You are a high-precision Browser Automation Agent.

### 🧠 INTELLIGENCE STRATEGY:

1. **DETERMINISTIC MODE (FAST - Use `execute_chain`):**
   - **Trigger:** When the path is standard (e.g., "Search YouTube for X", "Go to Amazon and search for Y", "Log in").
   - **Protocol:** You MUST bundle actions into a single `execute_chain` call.
   - **Standard Search Pattern:** `navigate` -> `wait(2s)` -> `click_visual(search_bar)` -> `type_text` -> `press_key(Enter)` -> `wait(2s)` -> `click_visual(first_result)`.
   - **Rule:** Always add a `wait` action after navigation or hitting Enter.

2. **REASONING MODE (ACCURATE - Step-by-Step):**
   - **Trigger:** When you need to find specific data, compare prices, or the element location is unknown until you see the page.
   - **Protocol:** Do NOT use chains. Execute ONE tool, stop, and analyze the new screenshot.
   - **Example:** "Find the cheapest red shoes". Click -> Wait -> Analyze -> Click next.

3. **INPUT & NAVIGATION:**
   - **Typing:** Always click the field *before* typing.
   - **Google Sheets:** Use Arrow Keys (`press_key`) instead of clicking small cells.
   - **Stuck?** If the page looks blank, use `navigate` to reload or `wait`.

4. **COORDINATES:**
   - The screen is a 1000x1000 grid. 0,0 is Top-Left.
   - Aim for the **center** of elements.
"""


class BrowserRelay:
    def __init__(self):
        self.ws = None
        self.msg_id = 0
        self.pending_requests = {}
        # Default metrics, updated dynamically
        self.metrics = {"w": 1280, "h": 720, "deviceScaleFactor": 1}

    @property
    def is_connected(self):
        return self.ws is not None and not self.ws.closed

    async def send_cdp(self, method: str, params: dict = None):
        if not self.is_connected:
            print("⚠️ CDP Error: Browser not connected.")
            return None

        self.msg_id += 1
        curr_id = self.msg_id
        future = asyncio.get_running_loop().create_future()
        self.pending_requests[curr_id] = future

        payload = {
            "id": curr_id,
            "method": "forwardCDPCommand",
            "params": {"method": method, "params": params or {}}
        }

        try:
            await self.ws.send_str(json.dumps(payload))
            return await asyncio.wait_for(future, timeout=TIMEOUT_CDP)
        except asyncio.TimeoutError:
            print(f"⚠️ CDP Timeout: {method}")
            self.pending_requests.pop(curr_id, None)
            return None
        except Exception as e:
            print(f"⚠️ CDP Error: {e}")
            self.pending_requests.pop(curr_id, None)
            return None

    async def update_metrics(self):
        """Forces an update of the window dimensions for accurate clicking."""
        res = await self.send_cdp("Runtime.evaluate", {
            "expression": "({w: window.innerWidth, h: window.innerHeight, dsf: window.devicePixelRatio})",
            "returnByValue": True
        })
        if res and 'result' in res and 'value' in res['result']:
            val = res['result']['value']
            self.metrics = {
                "w": val.get('w', 1280),
                "h": val.get('h', 720),
                "deviceScaleFactor": val.get('dsf', 1)
            }

    async def get_current_url(self):
        res = await self.send_cdp("Runtime.evaluate", {"expression": "window.location.href"})
        url = res['result']['value'] if res and 'result' in res else "unknown"
        browser_state.latest_url = url
        return url

    async def capture_screenshot_bytes(self):
        # Always update metrics before screenshot to ensure click math matches image
        await self.update_metrics()

        res = await self.send_cdp("Page.captureScreenshot", {
            "format": "jpeg",
            "quality": 100,
            "optimizeForSpeed": True
        })
        if not res or 'data' not in res: return None
        return base64.b64decode(res['data'])

    async def wait_for_load(self):
        """Robust wait for DOM to be ready."""
        for _ in range(15):
            res = await self.send_cdp("Runtime.evaluate", {"expression": "document.readyState"})
            if res and res.get('result', {}).get('value') == 'complete':
                break
            await asyncio.sleep(0.5)


relay = BrowserRelay()


# --- ROBUST LOW-LEVEL ACTIONS ---

async def send_enter_key():
    """
    Sends a ROBUST Enter key sequence.
    Essential for Google Sheets, Forms, and Single Page Apps.
    """
    # 1. Raw Key Down
    await relay.send_cdp("Input.dispatchKeyEvent", {
        "type": "rawKeyDown", "windowsVirtualKeyCode": 13, "code": "Enter", "key": "Enter",
        "text": "\r", "unmodifiedText": "\r"
    })
    # 2. Char Event (Triggers input submission)
    await relay.send_cdp("Input.dispatchKeyEvent", {
        "type": "char", "text": "\r"
    })
    # 3. Key Up
    await relay.send_cdp("Input.dispatchKeyEvent", {
        "type": "keyUp", "windowsVirtualKeyCode": 13, "code": "Enter", "key": "Enter"
    })


# --- HIGH-LEVEL TOOLS ---

async def exec_navigate(url: str):
    print(f"  ➡️ Navigating to: {url}")
    await relay.send_cdp("Page.navigate", {"url": url})
    await relay.wait_for_load()
    await asyncio.sleep(1.5)  # Extra buffer for rendering
    return f"Navigated to {url}"


async def exec_click_visual(x: int, y: int):
    # 1. Update Metrics to handle window resizing
    await relay.update_metrics()

    vw = relay.metrics.get('w', 1280)
    vh = relay.metrics.get('h', 720)

    # 2. Calculate Exact Pixel
    fx = int((x / 1000) * vw)
    fy = int((y / 1000) * vh)

    # 3. Clamp coordinates (Prevents CDP crashes if point is slightly off-screen)
    fx = max(0, min(fx, vw - 1))
    fy = max(0, min(fy, vh - 1))

    # 4. Inject Visual Feedback (Red Dot)
    dot_script = f"""
    (function() {{
        const d = document.createElement('div');
        d.style.cssText = 'position:fixed;left:{fx}px;top:{fy}px;width:12px;height:12px;background:red;border:2px solid white;border-radius:50%;z-index:2147483647;pointer-events:none;transform:translate(-50%,-50%);box-shadow:0 0 4px rgba(0,0,0,0.5);';
        document.body.appendChild(d);
        setTimeout(() => d.remove(), 1000);
    }})();
    """
    await relay.send_cdp("Runtime.evaluate", {"expression": dot_script})

    # 5. Robust Mouse Click (Press -> Wait -> Release)
    await relay.send_cdp("Input.dispatchMouseEvent",
                         {"type": "mousePressed", "x": fx, "y": fy, "button": "left", "clickCount": 1})
    await asyncio.sleep(0.1)  # Essential for JS listeners
    await relay.send_cdp("Input.dispatchMouseEvent",
                         {"type": "mouseReleased", "x": fx, "y": fy, "button": "left", "clickCount": 1})

    return f"Clicked {x},{y}"


async def exec_type_text(text: str):
    # 1. Pre-processing
    text = text.replace("\\n", "\n").replace("\\t", "\t")

    # 2. CLEAR FIELD LOGIC (Robustness)
    # Only clear if we aren't starting with a control char (like just hitting enter)
    if text and text[0] not in ("\n", "\t", "\r"):
        safe_select_script = """
            (function() {
                const el = document.activeElement;
                if (el) {
                    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
                        el.select();
                    } else if (el.isContentEditable) {
                        document.execCommand('selectAll', false, null);
                    }
                }
            })();
            """
        await relay.send_cdp("Runtime.evaluate", {"expression": safe_select_script})

    # 3. Type Character by Character
    typed_log = []
    for char in text:
        if char == "\t":
            await relay.send_cdp("Input.dispatchKeyEvent",
                                 {"type": "rawKeyDown", "windowsVirtualKeyCode": 9, "code": "Tab", "key": "Tab"})
            await relay.send_cdp("Input.dispatchKeyEvent",
                                 {"type": "keyUp", "windowsVirtualKeyCode": 9, "code": "Tab", "key": "Tab"})
            typed_log.append("[Tab]")
            await asyncio.sleep(0.05)

        elif char == "\n" or char == "\r":
            await send_enter_key()
            typed_log.append("[Enter]")
            await asyncio.sleep(0.2)  # Allow form submission to start

        else:
            await relay.send_cdp("Input.dispatchKeyEvent", {"type": "keyDown", "text": char, "unmodifiedText": char})
            await relay.send_cdp("Input.dispatchKeyEvent", {"type": "keyUp", "text": char, "unmodifiedText": char})
            typed_log.append(char)
            await asyncio.sleep(0.015)  # Natural typing speed

    return f"Typed: {''.join(typed_log)}"


async def exec_scroll(direction: str = "down"):
    # Hybrid Approach: Mouse Wheel + Keyboard
    viewport_w = relay.metrics.get('w', 1280)
    viewport_h = relay.metrics.get('h', 720)

    delta_y = 700 if direction == "down" else -700

    # 1. Mouse Wheel
    await relay.send_cdp("Input.dispatchMouseEvent", {
        "type": "mouseWheel", "x": viewport_w // 2, "y": viewport_h // 2, "deltaX": 0, "deltaY": delta_y
    })

    # 2. Keyboard PageDown/PageUp (Backup for documents)
    key = "PageDown" if direction == "down" else "PageUp"
    code = 34 if direction == "down" else 33

    await relay.send_cdp("Input.dispatchKeyEvent",
                         {"type": "rawKeyDown", "windowsVirtualKeyCode": code, "key": key, "code": key})
    await relay.send_cdp("Input.dispatchKeyEvent",
                         {"type": "keyUp", "windowsVirtualKeyCode": code, "key": key, "code": key})

    await asyncio.sleep(0.5)
    return f"Scrolled {direction}"


async def exec_chain(actions: list):
    """
    Robust Chain Executor.
    Maps generic params to specific robust functions.
    """
    log = []
    print(f"  ⛓️ Chain: {len(actions)} steps")

    for action in actions:
        name = action.get('name')
        params = action.get('params', {})

        if name == 'click_visual':
            await exec_click_visual(params['x'], params['y'])
            # Wait for click to settle
            await asyncio.sleep(0.3)

        elif name == 'type_text':
            await exec_type_text(params['text'])

        elif name == 'press_key':
            k = params.get('key', 'Enter')
            if k.lower() == 'enter':
                await send_enter_key()
            else:
                # Fallback for other keys
                await relay.send_cdp("Input.dispatchKeyEvent",
                                     {"type": "rawKeyDown", "windowsVirtualKeyCode": 0, "key": k, "code": k})
                await relay.send_cdp("Input.dispatchKeyEvent",
                                     {"type": "keyUp", "windowsVirtualKeyCode": 0, "key": k, "code": k})

        elif name == 'scroll':
            await exec_scroll(params.get('direction', 'down'))

        elif name == 'wait':
            await asyncio.sleep(params.get('seconds', 1))

        log.append(name)
        # Small delay between chain steps to prevent race conditions
        await asyncio.sleep(0.2)

    return f"Chain Executed: {', '.join(log)}"


# --- TOOL MAPPING ---
TOOL_MAP = {
    "navigate": exec_navigate,
    "click_visual": exec_click_visual,
    "type_text": exec_type_text,
    "execute_chain": exec_chain,
    "scroll": exec_scroll,
    "wait": lambda seconds=2: asyncio.sleep(seconds)
}

# --- GEMINI CONFIG ---
tools_config = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(name="navigate", description="Go to URL",
                                  parameters=types.Schema(type="OBJECT", properties={"url": {"type": "STRING"}},
                                                          required=["url"])),
        types.FunctionDeclaration(name="click_visual", description="Click coordinate (0-1000)",
                                  parameters=types.Schema(type="OBJECT", properties={"x": {"type": "INTEGER"},
                                                                                     "y": {"type": "INTEGER"}},
                                                          required=["x", "y"])),
        types.FunctionDeclaration(name="type_text", description="Type text (handles clearing field and Enter)",
                                  parameters=types.Schema(type="OBJECT", properties={"text": {"type": "STRING"}},
                                                          required=["text"])),
        types.FunctionDeclaration(name="execute_chain", description="Execute sequence: click, type, press_key",
                                  parameters=types.Schema(type="OBJECT", properties={"actions": {"type": "ARRAY",
                                                                                                 "items": {
                                                                                                     "type": "OBJECT",
                                                                                                     "properties": {
                                                                                                         "name": {
                                                                                                             "type": "STRING"},
                                                                                                         "params": {
                                                                                                             "type": "OBJECT"}}}}},
                                                          required=["actions"])),
        types.FunctionDeclaration(name="scroll", description="Scroll page", parameters=types.Schema(type="OBJECT",
                                                                                                    properties={
                                                                                                        "direction": {
                                                                                                            "type": "STRING",
                                                                                                            "enum": [
                                                                                                                "up",
                                                                                                                "down"]}},
                                                                                                    required=[
                                                                                                        "direction"])),
        types.FunctionDeclaration(name="task_done", description="Finish task",
                                  parameters=types.Schema(type="OBJECT", properties={"summary": {"type": "STRING"}})),
    ])
]


# --- MAIN LOOP ---

async def run_agent(user_prompt: str, status_callback=None):
    browser_state.is_running = True
    browser_state.stop_requested = False
    browser_state.user_feedback = None

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    history = [types.Content(role="user", parts=[types.Part.from_text(text=f"Task: {user_prompt}")])]
    final_summary = "Terminated"

    print(f"🤖 Agent Started: {user_prompt}")

    try:
        for step in range(50):
            # 1. Check Intervention
            if browser_state.stop_requested:
                final_summary = "🛑 Task stopped by user."
                break

            if browser_state.user_feedback:
                print(f"  👂 Feedback: {browser_state.user_feedback}")
                history.append(types.Content(role="user", parts=[
                    types.Part.from_text(text=f"USER INSTRUCTION: {browser_state.user_feedback}")]));
                browser_state.user_feedback = None

            # 2. Update Context
            url = await relay.get_current_url()
            img_bytes = await relay.capture_screenshot_bytes()

            if status_callback and img_bytes:
                await status_callback(f"Step {step} | {url[-30:]}", img_bytes)

            # 3. Prompt
            parts = [types.Part.from_text(text=f"Step {step}. URL: {url}")]
            if img_bytes:
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
            else:
                parts.append(types.Part.from_text(text="[Screenshot Failed]"))

            # Pruning
            if len(history) > 8:
                history = [history[0]] + history[-7:]

            history.append(types.Content(role="user", parts=parts))

            # 4. Inference
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=MODEL_ID,
                contents=history,
                config=types.GenerateContentConfig(tools=tools_config, system_instruction=SYSTEM_INSTRUCTION,
                                                   temperature=0.1)
            )

            if not response.candidates:
                await asyncio.sleep(2)
                continue

            model_msg = response.candidates[0].content
            history.append(model_msg)

            # 5. Execution
            if response.function_calls:
                tool_outputs = []
                for call in response.function_calls:
                    print(f"  🔧 {call.name}")
                    if call.name == "task_done":
                        final_summary = call.args.get('summary', 'Done')
                        browser_state.is_running = False
                        return final_summary

                    func = TOOL_MAP.get(call.name)
                    res = "Tool not found"
                    if func:
                        try:
                            res = await func(**call.args)
                        except Exception as e:
                            res = f"Error: {e}"

                    tool_outputs.append(types.Part.from_function_response(name=call.name, response={"result": res}))

                history.append(types.Content(role="tool", parts=tool_outputs))

            await asyncio.sleep(1.0)

    except Exception as e:
        final_summary = f"Crash: {e}"
        print(f"Loop Error: {e}")

    browser_state.is_running = False
    return final_summary


# --- SERVER ---
async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    relay.ws = ws
    print("🔌 Browser Connected")
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if "id" in data and data["id"] in relay.pending_requests:
                    relay.pending_requests[data["id"]].set_result(data.get("result"))
    finally:
        print("🔌 Browser Disconnected")
        relay.ws = None
    return ws


async def start_browser_server():
    app = web.Application()
    app.add_routes([web.get('/extension', ws_handler)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', PORT)
    await site.start()
    print(f"🚀 Server running on port {PORT}...")