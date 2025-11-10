# Use a lightweight base image
FROM python:3.12.3-bullseye

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app
COPY . .

# Install uvicorn (if not in requirements)
RUN pip install "uvicorn[standard]"

# Expose port
EXPOSE 8000

# Command for hot reload
CMD ["uvicorn", "career_assistant.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]