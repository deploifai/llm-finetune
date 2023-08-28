FROM python:3.10-slim

WORKDIR llama/try3/

COPY draft-3 draft-3

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY app.py .

CMD [ "python", "app.py" ]