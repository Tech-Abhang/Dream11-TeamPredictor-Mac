# 1. Use Python 3.11 runtime
FROM python:3.11-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container at /app
COPY req.txt .

# 4. Install any needed packages specified in req.txt
# Using --no-cache-dir to keep the image size down
RUN pip install --no-cache-dir -r req.txt

# 5. Copy the rest of the application files into the container at /app
# This includes app.py, the CSV data, and the ML model (which you will replace)
COPY app.py .
COPY main_file_updated.csv .
COPY ml_avg_predictor_model.pkl .

# 6. Create the data directory for potential input file mounting
RUN mkdir /app/data

# 7. Make port 8501 available to the world outside this container (Streamlit default)
EXPOSE 8501

# 8. Define environment variable for Streamlit (Fixed format)
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# 9. Run app.py when the container launches
# Use the shell form to handle Streamlit correctly
CMD ["streamlit", "run", "app.py"]