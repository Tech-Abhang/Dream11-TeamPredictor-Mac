# 1. Use Python 3.10 runtime
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file into the container at /app
COPY req.txt .

# 5. Install any needed packages specified in req.txt
# Using --no-cache-dir to keep the image size down
RUN pip install --no-cache-dir -r req.txt

# 6. Copy the rest of the application files into the container at /app
COPY app1.py .
COPY main_file_updated.csv .
COPY model.pkl .

# 7. Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOME=/root

# 8. Create an entrypoint script that sets up environment
RUN echo '#!/bin/bash\n\
# Run the application\n\
python3 app1.py "$@"\n\
' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# 9. Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]