FROM python:3.7-alpine3.14
WORKDIR /app

COPY . /app

RUN ls

RUN pip install Pipfile

CMD python manage.py runserver