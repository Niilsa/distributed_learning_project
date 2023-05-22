# Use the official TensorFlow image as the base image
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /app

# Copy the files from your host machine to the container's working directory
COPY . /app

# Install any necessary dependencies or packages
# (if needed for your specific project)

# Specify the command to run when the container starts
CMD [ "python", "your_script.py" ]

