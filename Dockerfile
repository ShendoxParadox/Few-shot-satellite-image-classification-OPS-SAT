FROM python:3.9
WORKDIR /app
COPY . /app
# Install python requirements
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --default-timeout=100
CMD ["/bin/bash"]
