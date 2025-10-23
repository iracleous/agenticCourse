"""
Example 2:
Goal: 
Realistic Example – “Research Weather and Send Email”

What This Demonstrates
    LLM reasoning → Planning
    Tool execution → Action
    LLM evaluation → Reflection
    Closed loop → Autonomy

"""

#setup
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
import requests
import smtplib
from email.mime.text import MIMEText

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


#define tools
def get_weather(city):
    """Simulated weather API call"""
    print(f"🔍 Checking weather for {city}...")
    # Replace with a real API like OpenWeatherMap
    weather_data = {"Paris": {"temp": 19, "condition": "Cloudy"}}
    w = weather_data.get(city, {"temp": "unknown", "condition": "unknown"})
    return f"The weather in {city} is {w['condition']} with {w['temp']}°C."

def send_email(recipient, subject, message):
    """Simulated email sending (no real SMTP here for safety)"""
    print(f"📧 Sending email to {recipient} with subject '{subject}'...")
    print(f"Message:\n{message}")
    return "Email sent successfully."


# LLM Prompts for Plan / Reflection

plan_prompt = ChatPromptTemplate.from_template("""
You are an autonomous agent. Your goal is: {goal}
Break this into a short step-by-step plan using available tools:
- get_weather(city)
- send_email(recipient, subject, message)
Return the plan as plain text steps.
""")

reflect_prompt = ChatPromptTemplate.from_template("""
You executed the following plan and got the result: {result}
Did you accomplish the goal: "{goal}"?
Reply with YES if complete or NO if more steps are needed.
""")

# Agent Loop

goal = "Research current weather in Paris and send an email report to user@example.com"
done = False

while not done:
    # --- Step 1: Ask LLM for a plan ---
    plan = llm.invoke(plan_prompt.format(goal=goal)).content
    print("\n🧠 Plan:\n", plan)

    # --- Step 2: Execute plan ---
    result = ""
    if "get_weather" in plan:
        weather_info = get_weather("Paris")
        result += weather_info + "\n"
    if "send_email" in plan:
        result += send_email("user@example.com", "Weather Report", weather_info)

    print("\n⚙️ Result:\n", result)

    # --- Step 3: Reflect using LLM ---
    reflection = llm.invoke(reflect_prompt.format(goal=goal, result=result)).content
    print("\n💭 Reflection:", reflection)

    if "YES" in reflection.upper():
        print("\n✅ Goal achieved. Stopping agent.")
        done = True
    else:
        print("\n🔁 Revising plan...\n")
