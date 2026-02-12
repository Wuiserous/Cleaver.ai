import asyncio
import json
import base64
import random
import os
import time
from aiohttp import web, WSMsgType
from google import genai
from google.genai import types

# --- CONFIGURATION ---
PORT = 18792
MODEL_ID = "gemini-3-flash-preview"
TIMEOUT_CDP = 20.0  # Increased for stability

SYSTEM_INSTRUCTION = """
You are a high-precision Browser Automation Agent.
Your goal: Complete the task in as few steps as possible.

CRITICAL PROTOCOLS:
1. **BATCHING:** Use `execute_chain` for interactions involving multiple steps (e.g., Search).
   - Pattern: Click input -> Type text -> Press Enter.
2. **SHOPPING/FORMS:** 
   - If a button/element is not visible, use `scroll`.
   - Once you click a final submission button (e.g., "Add to Cart", "Submit"), immediately call `task_done`.
3. **NAVIGATION:**
   - If the page looks wrong or stuck, try navigating again or checking the URL.
4. **GOOGLE SHEET**
    - When working in a google sheet, or excel sheet go column by column not row by row, complete one column first then go to the second column.
    - heavily rely or shortcuts like keyboard navigation, or formulas or existing google sheet or excel sheet tools according to the task.
    - use native tools of google sheet or excel sheets to make the task easily achievable. 

Output Format: Always provide a comprehensive summary when calling `task_done`.
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

    async def get_current_url(self):
        res = await self.send_cdp("Runtime.evaluate", {"expression": "window.location.href"})
        return res['result']['value'] if res and 'result' in res else "unknown"

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

    async def capture_screenshot_bytes(self):
        await self.update_metrics()
        res = await self.send_cdp("Page.captureScreenshot", {
            "format": "jpeg",
            "quality": 70,  # Increased quality for better text/coordinate recognition
            "optimizeForSpeed": True
        })
        if not res or 'data' not in res: return None
        return base64.b64decode(res['data'])

    async def wait_for_load(self):
        """Waits for the DOM to be ready."""
        for _ in range(10):
            res = await self.send_cdp("Runtime.evaluate", {"expression": "document.readyState"})
            if res and res.get('result', {}).get('value') == 'complete':
                break
            await asyncio.sleep(0.5)


relay = BrowserRelay()


# --- TOOLS ---

async def exec_navigate(url: str):
    print(f"  ➡️ Navigating to: {url}")
    await relay.send_cdp("Page.navigate", {"url": url})
    await relay.wait_for_load()
    await asyncio.sleep(1.0)  # Slight buffer for rendering
    return f"Navigated to {url}"


async def exec_click_visual(x: int, y: int):
    # Always update metrics before a coordinate calculation to handle resizing
    await relay.update_metrics()

    vw = relay.metrics.get('w', 1280)
    vh = relay.metrics.get('h', 720)

    # Calculate exact pixel
    fx = int((x / 1000) * vw)
    fy = int((y / 1000) * vh)

    # Clamp coordinates to stay within viewport (prevents CDP errors)
    fx = max(0, min(fx, vw - 1))
    fy = max(0, min(fy, vh - 1))

    # Visual Feedback (Red Dot) - Improved visibility
    dot_script = f"""
    (function() {{
        const d = document.createElement('div');
        d.style.cssText = 'position:fixed;left:{fx}px;top:{fy}px;width:12px;height:12px;background:red;border:2px solid white;border-radius:50%;z-index:2147483647;pointer-events:none;transform:translate(-50%,-50%);box-shadow:0 0 4px rgba(0,0,0,0.5);';
        document.body.appendChild(d);
        setTimeout(() => d.remove(), 1500);
    }})();
    """
    await relay.send_cdp("Runtime.evaluate", {"expression": dot_script})

    # Robust Click Sequence (Press -> Wait -> Release)
    # Increased wait time to 100ms ensures JS listeners capture the event
    await relay.send_cdp("Input.dispatchMouseEvent",
                         {"type": "mousePressed", "x": fx, "y": fy, "button": "left", "clickCount": 1})
    await asyncio.sleep(0.1)
    await relay.send_cdp("Input.dispatchMouseEvent",
                         {"type": "mouseReleased", "x": fx, "y": fy, "button": "left", "clickCount": 1})

    return f"Clicked at coordinates {x}, {y}"


async def exec_type_text(text: str):
    # 1. Detect and strip newline to prevent it from being typed as a character
    should_submit = False
    if "\n" in text:
        should_submit = True
        text = text.replace("\n", "")  # Clean the text

    # 2. Select all text first so we overwrite
    safe_select_script = """
        (function() {
            const el = document.activeElement;
            if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) {
                // If it's a standard input, use .select()
                if (typeof el.select === 'function') {
                    el.select();
                } else {
                    // Fallback for rich text editors (contentEditable div)
                    document.execCommand('selectAll', false, null);
                }
            }
        })();
        """
    await relay.send_cdp("Runtime.evaluate", {"expression": safe_select_script})

    # 3. Type the sanitized text
    for char in text:
        await relay.send_cdp("Input.dispatchKeyEvent", {"type": "keyDown", "text": char, "unmodifiedText": char})
        await relay.send_cdp("Input.dispatchKeyEvent", {"type": "keyUp", "text": char, "unmodifiedText": char})
        await asyncio.sleep(random.uniform(0.01, 0.04))

    # 4. Execute the Enter command safely if needed
    if should_submit:
        await asyncio.sleep(0.1)  # Wait for UI to catch up

        # Send raw Enter key (Code 13) WITHOUT 'text' property
        # This triggers the 'Submit' action rather than typing a new line
        await relay.send_cdp("Input.dispatchKeyEvent", {
            "type": "rawKeyDown",
            "windowsVirtualKeyCode": 13,
            "code": "Enter",
            "key": "Enter"
        })
        await asyncio.sleep(0.05)
        await relay.send_cdp("Input.dispatchKeyEvent", {
            "type": "keyUp",
            "windowsVirtualKeyCode": 13,
            "code": "Enter",
            "key": "Enter"
        })
        return f"Typed: {text} [Enter Pressed]"

    return f"Typed: {text}"


async def exec_scroll(direction: str = "down"):
    # Hybrid Approach: Mouse Wheel + Keyboard Arrow
    # This works on standard pages AND infinite scroll containers

    viewport_w = relay.metrics.get('w', 1280)
    viewport_h = relay.metrics.get('h', 720)
    x = viewport_w // 2
    y = viewport_h // 2

    delta_y = 700 if direction == "down" else -700

    print(f"  📜 Scrolling {direction}...")

    # 1. Mouse Wheel (Good for specific containers)
    await relay.send_cdp("Input.dispatchMouseEvent", {
        "type": "mouseWheel", "x": x, "y": y, "deltaX": 0, "deltaY": delta_y
    })

    # 2. Keyboard fallback (Good for document body scrolling)
    # "Page Down" (34) or "Page Up" (33)
    key_code = 34 if direction == "down" else 33
    key_name = "PageDown" if direction == "down" else "PageUp"

    await relay.send_cdp("Input.dispatchKeyEvent", {
        "type": "rawKeyDown", "windowsVirtualKeyCode": key_code, "key": key_name, "code": key_name
    })
    await relay.send_cdp("Input.dispatchKeyEvent", {
        "type": "keyUp", "windowsVirtualKeyCode": key_code, "key": key_name, "code": key_name
    })

    await asyncio.sleep(0.8)  # Wait for animation
    return f"Scrolled {direction}"


async def exec_chain(actions: list):
    results = []
    print(f"  ⛓️ Executing chain of {len(actions)} actions...")

    for i, action in enumerate(actions):
        name = action.get('name')
        params = action.get('params', {})

        if name == "click_visual":
            # If clicking an input, we want to ensure focus is settled
            res = await exec_click_visual(params['x'], params['y'])
            await asyncio.sleep(0.2)
        elif name == "type_text":
            res = await exec_type_text(params['text'])
        elif name == "scroll":
            res = await exec_scroll(params.get('direction', 'down'))
        elif name == "wait":
            wait_time = params.get('seconds', 1)
            await asyncio.sleep(wait_time)
            res = f"Waited {wait_time}s"
        elif name == "press_key":
            key_type = params.get('key', 'Enter')
            if key_type.lower() == 'enter':
                # Robust Enter Sequence
                await relay.send_cdp("Input.dispatchKeyEvent", {
                    "type": "rawKeyDown", "windowsVirtualKeyCode": 13,
                    "unmodifiedText": "\r", "text": "\r", "code": "Enter", "key": "Enter"
                })
                await relay.send_cdp("Input.dispatchKeyEvent", {"type": "char", "text": "\r"})
                await asyncio.sleep(0.05)
                await relay.send_cdp("Input.dispatchKeyEvent", {
                    "type": "keyUp", "windowsVirtualKeyCode": 13,
                    "unmodifiedText": "\r", "text": "\r", "code": "Enter", "key": "Enter"
                })
                res = "Pressed Enter"
            else:
                res = "Unknown Key"
        else:
            res = "Unknown Action"

        results.append(res)

        # Dynamic pause: If this was a click or enter, wait slightly longer for UI reaction
        if name in ["click_visual", "press_key"]:
            await asyncio.sleep(0.5)
        else:
            await asyncio.sleep(0.1)

    return " | ".join(results)


TOOL_MAP = {
    "navigate": exec_navigate,
    "click_visual": exec_click_visual,
    "type_text": exec_type_text,
    "execute_chain": exec_chain,
    "scroll": exec_scroll,
    "wait": lambda seconds=2: asyncio.sleep(seconds)
}

# --- GEMINI TOOL CONFIG ---

tools_config = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(name="navigate", description="Go to URL",
                                  parameters=types.Schema(type="OBJECT", properties={"url": {"type": "STRING"}},
                                                          required=["url"])),
        types.FunctionDeclaration(name="click_visual", description="Click coordinate (0-1000)",
                                  parameters=types.Schema(type="OBJECT", properties={"x": {"type": "INTEGER"},
                                                                                     "y": {"type": "INTEGER"}},
                                                          required=["x", "y"])),
# Add this inside your tools_config list
types.FunctionDeclaration(
    name="type_text",
    description="Type text into the active field",
    parameters=types.Schema(
        type="OBJECT",
        properties={"text": {"type": "STRING"}},
        required=["text"]
    )
),
        types.FunctionDeclaration(name="execute_chain",
                                  description="Execute sequence: click, type, press_key(enter), etc",
                                  parameters=types.Schema(type="OBJECT",
                                                          properties={"actions": {"type": "ARRAY", "items": {
                                                              "type": "OBJECT",
                                                              "properties": {"name": {"type": "STRING"},
                                                                             "params": {"type": "OBJECT"}}
                                                          }}}, required=["actions"])),
        types.FunctionDeclaration(name="scroll", description="Scroll page",
                                  parameters=types.Schema(type="OBJECT", properties={
                                      "direction": {"type": "STRING", "enum": ["up", "down"]}},
                                                          required=["direction"])),
        types.FunctionDeclaration(name="task_done", description="Finish task",
                                  parameters=types.Schema(type="OBJECT", properties={"summary": {"type": "STRING"}})),
    ])
]


# --- AGENT LOOP ---

async def run_agent(user_prompt: str, status_callback=None):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    history = []
    history.append(types.Content(role="user", parts=[types.Part.from_text(text=f"Task: {user_prompt}")]))

    print(f"🤖 Agent Started: {user_prompt}")

    for step in range(30):
        url = await relay.get_current_url()

        if status_callback:
            await status_callback(f"Step {step}: Analyzing {url[:40]}...")

        # 1. Capture State
        img_bytes = await relay.capture_screenshot_bytes()

        # 2. History Management (Pruning)
        if len(history) > 4:
            oldest_retained = history[-3]
            if oldest_retained.role == "user":
                text_only = [p for p in oldest_retained.parts if p.text]
                if not text_only: text_only = [types.Part.from_text(text="(Previous step image removed)")]
                oldest_retained.parts = text_only

        # 3. Construct Message
        context_msg = f"Step {step} | URL: {url}"
        parts = [types.Part.from_text(text=context_msg)]

        if img_bytes:
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
        else:
            parts.append(types.Part.from_text(text="(Screenshot failed)"))

        history.append(types.Content(role="user", parts=parts))

        # 4. Generate Response
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=MODEL_ID,
                contents=history,
                config=types.GenerateContentConfig(
                    tools=tools_config,
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.0
                )
            )
        except Exception as e:
            print(f"❌ API Error: {e}")
            return f"Agent Crashed: {e}"

        # 5. Handle Response
        if not response.candidates:
            print("❌ No candidates returned.")
            return "Error: No response from model."

        model_msg = response.candidates[0].content
        history.append(model_msg)

        if response.function_calls:
            tool_responses = []
            for call in response.function_calls:
                if call.name == "task_done":
                    summary = call.args.get('summary')
                    print(f"✅ DONE: {summary}")
                    return summary

                print(f"🛠️ Tool: {call.name}")
                if status_callback: await status_callback(f"Step {step}: {call.name}...")

                func = TOOL_MAP.get(call.name)
                res_text = await func(**call.args) if func else "Error: Tool not found"

                tool_responses.append(types.Part.from_function_response(
                    name=call.name,
                    response={'result': res_text}
                ))

            history.append(types.Content(role="tool", parts=tool_responses))

            # Additional small wait after tool execution before taking next screenshot
            await asyncio.sleep(1.0)
        else:
            text_content = " ".join([p.text for p in model_msg.parts if p.text])
            print(f"🤔 Model thought: {text_content}")
            history.append(
                types.Content(role="user", parts=[types.Part.from_text(text="Proceed with the next action.")]))

    return "❌ Max steps reached."


# --- SERVER ---

async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    relay.ws = ws
    print("🔌 Browser Extension Connected")
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if "id" in data and data["id"] in relay.pending_requests:
                    relay.pending_requests[data["id"]].set_result(data.get("result"))
    except Exception as e:
        print(f"🔌 Connection Error: {e}")
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
    print(f"🚀 Server running on port {PORT}. Waiting for extension...")


async def execute_browser_task(task_prompt: str, status_callback=None):
    if not relay.is_connected:
        return "❌ Error: Browser extension not connected."
    return await run_agent(task_prompt, status_callback)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(start_browser_server())

    print("\n⏳ Waiting 5s for browser connection...")
    time.sleep(5)

    if relay.is_connected:
        loop.run_until_complete(execute_browser_task("Go to google.com and search for 'python asyncio'"))
    else:
        print("❌ Please connect the browser extension.")

    loop.run_forever()