"""
Example 4:
Goal: multi-agent version (e.g., “Research Agent” + “Communication Agent”)

"""


import requests
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

# --- Setup your Azure OpenAI credentials ---
# Make sure you have these set as environment variables or replace with strings
# os.environ["AZURE_OPENAI_API_KEY"] = "your-key"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-endpoint.openai.azure.com/"

import os
import dotenv as dt
dt.load_dotenv()

temperature  = 0
max_tokens = 150
llm = AzureChatOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    model_name = os.getenv("AZURE_OPENAI_MODEL"),
    max_tokens=max_tokens, 
    temperature=temperature
)
# --- Helper functions ---

def get_weather(city: str) -> str:
    """Mocked weather fetcher (you can replace with a real API call)."""
    print(f"🔍 Fetching weather for {city}...")
    # In real use, call a weather API like OpenWeatherMap here
    fake_weather = "20°C, clear sky"
    return f"The current weather in {city} is {fake_weather}."

def send_email(to: str, subject: str, body: str) -> str:
    """Mock email sender."""
    print(f"📧 Sending email to {to} with subject '{subject}'...")
    # Replace with actual email integration (e.g., SendGrid, SMTP)
    return f"Email sent to {to} with subject '{subject}'."

def done(reflection: str) -> bool:
    """Check if the LLM thinks the task is complete."""
    return "complete" in reflection.lower() or "done" in reflection.lower()

# --- Agentic loop ---
def agent_loop(goal: str):
    print(f"\n🎯 Goal: {goal}")

    while True:
        # Step 1: Ask LLM to create a plan
        plan_prompt = ChatPromptTemplate.from_template(
            "You are an autonomous AI agent. Your goal is: {goal}. "
            "Generate a short, clear plan of actions."
        )
        plan = llm.invoke(plan_prompt.format_messages(goal=goal))
        print(f"\n🧩 Plan:\n{plan.content}")

        # Step 2: Execute the plan
        result = ""
        if "weather" in plan.content.lower():
            result += get_weather("Paris") + "\n"
        if "email" in plan.content.lower():
            result += send_email("user@example.com", "Weather Update", "It’s sunny in Paris!") + "\n"

        print(f"\n⚙️ Result:\n{result}")

        # Step 3: Reflection and evaluation
        reflection_prompt = ChatPromptTemplate.from_template(
            "Given the goal '{goal}' and the result:\n{result}\n"
            "Reflect on whether the goal is complete or more steps are needed."
        )
        reflection = llm(reflection_prompt.format_messages(goal=goal, result=result))
        print(f"\n🪞 Reflection:\n{reflection.content}")

        if done(reflection.content):
            print("\n✅ Task completed.")
            break
        else:
            print("\n🔁 Revising plan...")

# --- Run the agent ---
if __name__ == "__main__":
    agent_loop("Research current weather in Paris and send an email")
