# CLEAVER: THE AUTONOMOUS UNIVERSAL AGENT

## OVERVIEW
Cleaver is a sophisticated, self-evolving AI ecosystem designed to act as a fully autonomous personal assistant. Unlike standard chatbots, Cleaver possesses "eyes" to see the web, "hands" to execute code, and a "memory" that grows with every interaction. It is specifically engineered to handle complex, multi-step digital tasks while maintaining a proactive presence in the user's life.

## CORE ARCHITECTURE
The system is built on a three-pillar architecture that separates intelligence, action, and visual perception.

| COMPONENT | PRIMARY RESPONSIBILITY |
| :--- | :--- |
| **THE ORCHESTRATOR** | Telegram Interface, Gemini Logic, Location Tracking |
| **THE TOOLBOX** | Python Execution, Skill Library, Proactive Tasks |
| **THE VISUAL AGENT** | Autonomous Browsing, CDP Relay, Visual Reasoning |

### THE ORCHESTRATOR (MAIN.PY)
The Orchestrator serves as the central nervous system of Cleaver. It manages the primary communication channel via Telegram and coordinates the flow of information between the user and the Gemini AI model.
- **MULTIMODAL INTERACTION**: Processes text, images, and voice notes seamlessly.
- **LIVE LOCATION AWARENESS**: Silently tracks the user's GPS coordinates and converts them into readable addresses.
- **SECURITY INTERCEPTOR**: Identifies sensitive requests (like API keys) and requests them securely.

### THE TOOLBOX (MY_TOOLS.PY)
The Toolbox provides Cleaver with the ability to interact with the physical and digital world.
- **UNIVERSAL PYTHON EXECUTOR**: Allows the agent to write and run Python scripts to solve any problem.
- **SKILLS LIBRARY**: A self-evolution mechanism where the agent saves successful scripts as "Skills."
- **PROACTIVE HEARTBEAT**: A background scheduler that "wakes up" the agent every 25 minutes.
- **LONG-TERM MEMORY**: Maintains a persistent Markdown-based memory file.

### THE VISUAL AGENT (BROWSER_AGENT.PY)
The Visual Agent gives Cleaver the ability to navigate the web exactly like a human.
- **VISUAL REASONING**: Uses screenshots and visual AI to "see" buttons, forms, and data.
- **DETERMINISTIC CHAINS**: Executes high-speed chains of actions for common tasks.
- **REASONING MODE**: Performs step-by-step analysis for complex tasks.

## SYSTEM PHILOSOPHY
Cleaver is designed to be **Autonomous** and **Universal**. It does not just suggest actions; it performs them. Through the combination of Python execution and browser automation, there is virtually no digital task Cleaver cannot attempt.

## SECURITY & PRIVACY
Cleaver operates with a "Security First" mindset. Sensitive credentials are intercepted at the interface level and never passed to the AI model. All code execution happens in a controlled local environment.
