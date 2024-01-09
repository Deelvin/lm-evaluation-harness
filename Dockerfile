FROM python:3.9.18-slim

COPY . .

RUN python3 -m pip install -e . \
    && python3 -m pip install pandas

ENTRYPOINT [ "python3", "-u", "./run.py" ]