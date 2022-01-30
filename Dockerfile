FROM python:3.8-slim

COPY . /root

WORKDIR /root/web_interface

RUN pip install -r ../requirements.txt
