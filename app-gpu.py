import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "./v3"
TOKENIZER_PATH = "./v3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load or cache the model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
except OSError:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(TOKENIZER_PATH)
    model.save_pretrained(MODEL_PATH)

model.to(device)

def generate_response(user_input: str):
    print("received user input: " + user_input)

    input_context = "you are an AI model that provides physics questions\n###Human:" + user_input + "\n###Assistant:"

    input_ids = tokenizer.encode(input_context, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=400, temperature=0.4, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("generated text: " + generated_text)

    return generated_text

def run_gradio_interface():
    # Gradio interface for the model
    interface = gr.Interface(
        fn=generate_response,
        inputs=gr.Textbox(label="Input", placeholder="Type your query"),
        outputs=gr.Textbox(label="Response"),
        title="Llama 2 physics helper",
    )
    interface.launch(server_name="0.0.0.0", inline=False)

# Load the model and tokenizer at program startup
if __name__ == "__main__":
    run_gradio_interface()