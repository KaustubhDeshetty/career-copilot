"""
test_router.py
Standalone router test -- no pipelines or agents required.
Run this after router.py to sanity-check routing decisions before
wiring orchestrator.py to real Researcher/Writer/Critic logic.
"""

import os
from groq import Groq
from router import route_request

client = Groq(api_key=os.environ["GROQ_API_KEY"])

test_prompts = [
    # Should route to apply
    "Here's a job posting: Software Engineer at TechCorp, needs 3 years "
    "Python and ML experience. Can you help me tailor my resume bullets?",

    # Should route to interview
    "Quiz me on decision trees, I have a DS interview next week.",

    # Should route to learn
    "Can you help me review TF-IDF today? I keep forgetting how it works.",

    # Ambiguous -- this is the one to watch closely
    "Help me with data science.",

    # Mentions two modes at once -- also worth watching
    "I have an interview next week for a job I found, can you help me prep?",
]

print("TESTING ROUTER IN ISOLATION\n" + "=" * 55)
for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    mode, args = route_request(prompt, client)
    print(f"  -> mode: {mode}")
    print(f"  -> args: {args}")
    assert mode in {"apply", "interview", "learn"}, f"Unexpected mode: {mode}"

print("\nAll prompts routed to a valid mode.")
