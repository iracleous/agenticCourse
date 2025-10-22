"""
Ex3. Reactive agent using LiteLLM
Goal: Demonstrate a simple reactive agent that interacts with a local LLM
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

# ----------------------------------------------------
# 1️⃣ Load environment variables
# ----------------------------------------------------
load_dotenv()

OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

# ----------------------------------------------------
# 2️⃣ Initialize LiteLLM client (e.g., Ollama proxy or local server)
# ----------------------------------------------------
client = OpenAI(base_url=OPENAI_ENDPOINT, api_key=OPENAI_API_KEY)

# ----------------------------------------------------
# 3️⃣ Define a simple reactive agent
# ----------------------------------------------------
class ReactiveLLMAgent:
    """
    A simple reactive agent that perceives input, decides based on rules,
    and reacts using either built-in logic or a local LLM.
    """

    def __init__(self, llm_client, model):
        self.llm_client = llm_client
        self.model = model
        self.message = ""

    def perceive(self, message: str):
        """Process environment input (convert to lowercase for easier matching)."""
        self.message = message.lower()

    def act(self):
        """Immediate reaction (no memory or planning)."""
        if "weather" in self.message:
            return self._get_weather()
        elif "time" in self.message:
            return self._get_time()
        else:
            return self._ask_llm()

    def _get_weather(self):
        """Fake weather API (example placeholder)."""
        # You could replace this with a real API call if desired
        return "☀️ It’s sunny and 25°C."

    def _get_time(self):
        """Return current local time."""
        return f"🕒 The current time is {datetime.now().strftime('%H:%M:%S')}."

    def _ask_llm(self):
        """Ask the local model through LiteLLM-compatible client."""
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a concise and helpful assistant."},
                {"role": "user", "content": self.message}
            ]
        )
        return response.choices[0].message.content

# ----------------------------------------------------
# 4️⃣ Run the agent interactively
# ----------------------------------------------------
if __name__ == "__main__":
    agent = ReactiveLLMAgent(client, OPENAI_MODEL_NAME)
    print("🤖 Reactive LLM Agent ready (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        start = time.time()
        agent.perceive(user_input)
        response = agent.act()
        end = time.time()

        print(f"Agent: {response}")
        print(f"(⏱️ Response time: {end - start:.2f} seconds)\n")
