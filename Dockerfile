FROM python:3.10-slim

COPY . .

RUN python3 -m pip install -e . \
    && python3 -m pip install pandas \
    && export PYTHONPATH=$(pwd):$(pwd)/scripts

ENTRYPOINT [ "python3", "-u", "./scripts/run_endpoints_benchmark.py" ]