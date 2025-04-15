import discord
import os
import logging
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models

# Load environment variables from .env
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Discord bot intents
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Generation and safety configs (declared before function)
generation_config = {
    "max_output_tokens": 2048,
    "temperature": 1,
    "top_p": 1,
    "top_k": 32,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Function to generate response using Gemini
def generate_response_from_gemini(text):
    try:
        vertexai.init(project="ondc-project-cloud", location="asia-southeast1")
        model = GenerativeModel("gemini-1.0-pro-vision-001")
        responses = model.generate_content(
            [text],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True
        )
        full_response = ""
        for response in responses:
            if response.text:
                full_response += response.text
        return full_response if full_response else "No response generated."
    except Exception as e:
        logging.error(f"Error generating Gemini response: {e}")
        return "⚠️ Sorry, I ran into an issue generating a response."

# Triggered when bot is ready
@client.event
async def on_ready():
    logging.info(f"✅ Logged in as {client.user}")

# Triggered when a message is received
@client.event
async def on_message(message):
    logging.info("📥 Message received")
    
    # Ignore bot's own messages
    if message.author == client.user:
        return

    # Optional: Only respond if message starts with "!ask"
    if message.content.startswith("!ask"):
        user_message = message.content[5:].strip()
        logging.info(f"🔍 User Message: {user_message}")

        bot_reply = generate_response_from_gemini(user_message)
        logging.info(f"🤖 Bot Reply: {bot_reply}")

        # Send formatted response
        await message.channel.send(f"{message.author.mention}\n```{bot_reply}```")

# Run the bot
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    logging.error("❌ Discord token not found. Make sure DISCORD_TOKEN is set in the environment.")
else:
    client.run(DISCORD_TOKEN)
