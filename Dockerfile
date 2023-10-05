# Dockerfile
FROM python:3.8-slim

WORKDIR /app

# Copy the model training script 
COPY train_model.py .

# Install the requirements file for the docker
COPY requirements.txt .
RUN pip install -r requirements.txt

# Command to run the training script
CMD ["python", "train_model.py"]
