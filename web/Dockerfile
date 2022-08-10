FROM ubuntu:20.04

RUN apt update \
    && apt -y install python3-docker python3-pip

COPY config.json web/

COPY web web/
WORKDIR /web

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py"]