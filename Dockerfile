FROM tensorflow/tensorflow:latest-gpu-py3

# Copy the current directory contents into the container at /app
COPY ./requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip --no-cache-dir install -r requirements.txt

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run jupyter when container launches
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]
