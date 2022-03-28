FROM python:3.7-alpine3.14
WORKDIR /app

COPY . /app

RUN cd app && pip install

CMD python manage.py runserver