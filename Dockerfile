# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PORT=8080
ENV HOST=0.0.0.0

# Expose port 8080
EXPOSE 8080

# Command to run the application
CMD streamlit run --server.port $PORT --server.address $HOST app.py 