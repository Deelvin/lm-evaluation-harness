# Install Python 3.10
FROM python:3.10

# Install base dependencies
RUN apt-get update && \
    apt-get install -y build-essential libpq-dev

# Set root as lm-eval
WORKDIR /lm_eval

# Copy repository to container
COPY . .

# Copy requirements.txt to container (not neccessary)
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Install test requirements
RUN pip install pytest allure-pytest pytest-csv
RUN apt-get install -y jq