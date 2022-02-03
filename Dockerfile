FROM python:3.8-slim

COPY . /root

WORKDIR /root/web_interface

ENV FLASK_ENV=production
ENV FLASK_APP=run.py

ARG FLASK_SECRET_KEY
ENV FLASK_SECRET_KEY $FLASK_SECRET_KEY

RUN pip install -r ../requirements.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]