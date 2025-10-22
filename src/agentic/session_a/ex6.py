

# using @tools


import os
import json
import requests
from functools import wraps
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI


# 1️⃣ Load environment
load_dotenv()
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# 2️⃣ Connect to local LLM (LiteLLM)
client = OpenAI(base_url=OPENAI_ENDPOINT, api_key=OPENAI_API_KEY)



# 3️⃣ Global registry
TOOLS = {}

def tool(name=None, description=None):
    """
    Decorator to register a function as a tool with metadata.
    """
    def decorator(func):
        tool_name = name or func.__name__
        doc = description or func.__doc__ or "No description"
        TOOLS[tool_name] = {
            "func": func,
            "description": doc.strip()
        }
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

# 4️⃣ Define REAL tools
@tool(description="Get the current local time.")
def get_time():
    """Return the current time in local timezone."""
    return f"🕒 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."

@tool(description="Get the current weather for a city using OpenWeather API.")
def weather(city: str = "Athens"):
    """Fetch real weather data."""
    if not OPENWEATHER_API_KEY:
        return "⚠️ Missing OpenWeather API key."
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        r = requests.get(url)
        data = r.json()
        if data.get("cod") != 200:
            return f"⚠️ Weather not found for {city}."
        desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"☁️ {city.title()}: {desc}, {temp}°C"
    except Exception as e:
        return f"⚠️ Error fetching weather: {e}"

@tool(description="Get a random joke from the official JokeAPI.")
def joke():
    """Fetch a random programming joke."""
    try:
        r = requests.get("https://v2.jokeapi.dev/joke/Programming?type=single")
        data = r.json()
        return "😂 " + data.get("joke", "Couldn't get a joke.")
    except Exception as e:
        return f"⚠️ Error fetching joke: {e}"

@tool(description="Search a short Wikipedia summary for a topic.")
def wikipedia_search(topic: str):
    """Fetch a short summary from Wikipedia."""
    try:
        r = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}")
        data = r.json()
        if "extract" in data:
            return f"📚 {data['title']}: {data['extract']}"
        return "⚠️ No summary found."
    except Exception as e:
        return f"⚠️ Error: {e}"

# 5️⃣ Agent Class
class ToolAgent:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def decide_action(self, user_input):
        """Ask LLM which tool to use."""
        tool_descriptions = "\n".join(
            [f"- {name}: {meta['description']}" for name, meta in TOOLS.items()]
        )

        system_prompt = f"""
        You are a helpful AI assistant that can use tools.

        Available tools:
        {tool_descriptions}

        When user input matches a tool, respond **only** in JSON like:
        {{"tool": "weather", "args": {{"city": "Athens"}}}}

        Otherwise, respond in plain text.
     
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def execute(self, decision):
        """Run tool or return text."""
        try:
            data = json.loads(decision)
            tool_name = data.get("tool")
            args = data.get("args", {})
            if tool_name in TOOLS:
                func = TOOLS[tool_name]["func"]
                result = func(**args)
                return f"(Used {tool_name}) {result}"
        except json.JSONDecodeError:
            # Plain text response
            return decision
        except Exception as e:
            return f"⚠️ Tool error: {e}"

    def run(self, query):
        decision = self.decide_action(query)
        return self.execute(decision)

# 6️⃣ Interactive Loop
if __name__ == "__main__":
    agent = ToolAgent(client, OPENAI_MODEL_NAME)
    print("🤖 Smart Tool Agent ready. Type 'exit' to quit.\n")

    while True:
        user = input("You: ")
        if user.lower() in ["exit", "quit"]:
            break
        print("Agent:", agent.run(user), "\n")
