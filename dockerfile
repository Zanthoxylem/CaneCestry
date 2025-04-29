# ---------- CaneCestry Docker image ----------

# 1. Start from a small official Python image
FROM python:3.11-slim

# 2. Install system packages that some Python libs need
RUN apt-get update && apt-get install -y --no-install-recommends \
        graphviz gcc             \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory inside the container
WORKDIR /app

# 4. Copy only requirements first (better layer caching)
COPY requirements.txt .

# 5. Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your code and data
COPY . .

# 7. Expose the port Dash/Flask will listen on
EXPOSE 8050

# 8. Command to run when the container starts
CMD ["python", "flask_app.py"]
