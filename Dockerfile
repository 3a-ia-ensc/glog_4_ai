FROM python:3.7-alpine

RUN python -m pip install --upgrade pip

RUN pip install tensorflow-serving-api

COPY requirements.txt /
RUN pip install -r /requirements.txt
COPY src /app/src/data_processing.py
COPY src /app/src/model.py
COPY www /app/www

EXPOSE 5000
WORKDIR /app/www
CMD ["flask", "run", "--host=0.0.0.0"]