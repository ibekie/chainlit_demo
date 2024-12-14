import chainlit as cl
import openai
import os
import base64

# Get API keys from environment
api_key = os.getenv("RUNPOD_API_KEY")
runpod_serverless_id = os.getenv("RUNPOD_SERVERLESS_ID")

# Configure endpoint URL for RunPod
endpoint_url = f"https://api.runpod.ai/v2/{runpod_serverless_id}/openai/v1"

# Create OpenAI client with RunPod endpoint
client = openai.AsyncClient(api_key=api_key, base_url=endpoint_url)

# Model configuration
model_kwargs = {
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "temperature": 0.3,
    "max_tokens": 500
}

@cl.on_message
async def on_message(message: cl.Message):
    # Get message history from session
    message_history = cl.user_session.get("message_history", [])
    
    # Add system message if this is the first message
    if not message_history:
        message_history.append({
            "role": "system",
            "content": "You are Mistral-7B-Instruct-v0.3, a helpful AI assistant."
        })

    # Check for image attachments
    images = [file for file in message.elements if "image" in file.mime] if message.elements else []

    if images:
        # Switch to GPT-4 Vision model for image processing
        model_kwargs["model"] = "gpt-4-vision-preview"
        
        # Read and encode image
        with open(images[0].path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        
        # Add message with image to history
        message_history.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message.content if message.content else "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })
    else:
        # Regular text message
        message_history.append({"role": "user", "content": message.content})

    # Create empty message for streaming
    response_message = cl.Message(content="")
    await response_message.send()
    
    try:
        # Stream the response
        stream = await client.chat.completions.create(
            messages=message_history,
            stream=True,
            **model_kwargs
        )

        # Process streaming response
        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await response_message.stream_token(token)

        await response_message.update()

        # Add assistant's response to history
        message_history.append({"role": "assistant", "content": response_message.content})
        cl.user_session.set("message_history", message_history)

        # Reset model back to Mistral if we switched for image processing
        if images:
            model_kwargs["model"] = "mistralai/Mistral-7B-Instruct-v0.3"

    except Exception as e:
        print(f"Error: {str(e)}")
        await response_message.update(content=f"Error: {str(e)}")

@cl.on_chat_start
async def on_chat_start():
    """Send a welcome message when the chat starts."""
    await cl.Message(
        content="Hello! I'm your AI assistant. I can help with text responses and analyze images. How can I help you today?"
    ).send()