from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo

# export OPENAI_API_KEY='sk-HdGVyDNpTrJV0BxJ59B9l65hcO68OUJnUw8ElP3I6XT3BlbkFJdPt-m06qGCM9EbcwS9tkViZP1kRVGJkRq5B1t97d4A'


import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is None:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Now, you can use the API key for OpenAI calls



web_agent = Agent(
    name="Web Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)
web_agent.print_response("what is the Base salary of Apple CEO in 2024?", stream=True)
