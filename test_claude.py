import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "You are a PRD review assistant for a fintech product team. What are the 3 most important sections of a good PRD for a credit card feature?"
        }
    ]
)

print(message.content[0].text)
