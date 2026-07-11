# 1. Force the correct architecture and use a minimal Python base to save space
FROM --platform=linux/amd64 python:3.11-slim

# 2. Prevent Python from writing .pyc files and force stdout buffering (good for server logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy only the requirements first to leverage Docker caching
COPY requirements.txt .

# 5. Install dependencies without saving the massive pip cache
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your exact Track 2 logic files into the container
COPY main.py .
COPY video_utils.py .

# 7. Set the command that executes immediately when the automated judge boots the container
CMD ["python", "main.py"]