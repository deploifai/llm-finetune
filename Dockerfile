FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR llm

COPY v3 v3

# Install the dependencies
RUN pip install gradio transformers

# Copy the rest of the application code to the working directory
COPY app-gpu.py .

EXPOSE 7860

CMD [ "python", "app-gpu.py" ]