# using python 3.11 slim image for smaller size
FROM python:3.11-slim

# Setting the working directory inside container
WORKDIR /app

# Installing system dependencies needed for OpenCV, TesnorFlow, etc.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# coping the requirements file first for caching
COPY requirements.txt .

# Installing python dependencies
RUN pip install -r requirements.txt

# Downloading the NLTK data for sentiment analysis
RUN python -c "import nltk; nltk.download('vader_lexicon')"


# copying all project files to container
COPY . .

# creating the upload directory (for the uploads), although app.py creates it but to make sure it exists
RUN mkdir -p uploads

# exposing the port 5000 for Flask app
EXPOSE 5000

# command to run Flask app
CMD [ "python", "app.py" ]