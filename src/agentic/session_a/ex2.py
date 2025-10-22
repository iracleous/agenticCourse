"""
Ex2. Reactive agent
Goal:  Demonstrate a simple reactive agent that responds to user input based 
on predefined rules.
"""

# 

class ReactiveAgent:
    def __init__(self):
        # Define simple "if → then" rules
        self.rules = {
            "hello": "Hi there! How can I help?",
            "bye": "Goodbye!",
            "weather": "I can’t see the sky, but it might be sunny!",
            "hungry": "You should eat something healthy!"
        }

    def perceive(self, environment_input: str):
        """Perceive environment (user input)"""
        self.current_input = environment_input.lower()

    def act(self):
        """React immediately based on current input"""
        for keyword, response in self.rules.items():
            if keyword in self.current_input:
                return response
        return "I'm not sure how to respond to that."

# --- Usage ---
agent = ReactiveAgent()

while True:
    msg = input("You: ")
    if msg.lower() == "exit":
        break
    agent.perceive(msg)
    print("Agent:", agent.act())

