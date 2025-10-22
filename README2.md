AI researchers (and now LLM framework designers) usually classify agents into four or five main categories depending on their complexity and internal architecture.

1️⃣ Simple Reflex Agents (Reactive)

Behavior: purely rule-based “if → then”.
No memory or model.
Example: a thermostat turning on heating if temperature < 20°C.

2️⃣ Model-Based Reflex Agents (Reactive)

Keep an internal state (model) of the world.
Can handle partially observable environments.
Example: a self-driving car that remembers the last positions of other vehicles.

3️⃣ Goal-Based Agents (Planner / Orchestrator)

Choose actions to achieve specific goals.
They plan ahead.
Example: an AI planner that chooses a sequence of moves to reach a destination.

4️⃣ Utility-Based Agents (Decision optimizer)

Go beyond goals — they evaluate outcomes by assigning utilities (scores) to states.
Aim to maximize long-term satisfaction, not just reach a goal.
Example: a stock trading agent that optimizes expected profit.

5️⃣ Learning Agents (Self-improving agent / fine-tuned LLM)

Learn from past actions and feedback to improve over time.
Usually include a performance element (acting) and a learning element (improving).
Example: reinforcement learning systems like AlphaGo or adaptive LLM agents.
