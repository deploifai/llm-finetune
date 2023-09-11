import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained("./v3")
model = AutoModelForCausalLM.from_pretrained("./v3")

def results(user_input: str):
    input_context = "you are an ai model that provides physics questions\n ###Human:"+user_input+"\n###Assistant:"

    input_ids = tokenizer.encode(input_context, return_tensors="pt")

    output = model.generate(input_ids, max_length=400, temperature=0.4, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokenz=True)

    return generated_text


# gradio interface for the model
demo = gr.Interface(
    fn=results,
    inputs=gr.Textbox(label="Input", placeholder="Type your query"),
    outputs=gr.Textbox(label="Response"),
    title="Llama 2 physics helper",
)

demo.launch(server_name="0.0.0.0", inline=False)
