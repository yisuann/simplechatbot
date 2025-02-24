import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize chat history
chat_history_ids = None

def chat(user_input):
    global chat_history_ids

    # Exit condition
    if user_input.lower() == "exit":
        return "Goodbye!"

    # Tokenize the user input and add to chat history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)

    # Append the new user input to the chat history
    chat_history_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Truncate history if it exceeds the maximum length
    max_history_length = 500  # Adjust as needed
    if chat_history_ids.shape[-1] > max_history_length:
        chat_history_ids = chat_history_ids[:, -max_history_length:]

    # Generate a response from the model
    response_ids = model.generate(
        chat_history_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )

    # Decode the response
    bot_response = tokenizer.decode(response_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

    # Update chat history with the bot's response
    chat_history_ids = response_ids

    return bot_response

# Create a Gradio interface
iface = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="Chatbot",
    description="A simple chatbot using DialoGPT-medium.",
)

# Launch the app
iface.launch()