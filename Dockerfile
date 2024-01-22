FROM python:3.9
# Update package list and install Nano
RUN apt-get update && \
    apt-get install -y nano && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
# Install python requirements
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --default-timeout=100
CMD ["/bin/bash"]
