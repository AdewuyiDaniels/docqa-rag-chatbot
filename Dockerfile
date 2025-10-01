# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code from your host to your image filesystem.
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable for Streamlit
ENV STREAMLIT_SERVER.PORT 8501
ENV STREAMLIT_SERVER.ADDRESS 0.0.0.0

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]