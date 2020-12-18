FROM python:3.8

RUN python -m pip install --upgrade pip

COPY requirements.txt /
RUN pip install -r /requirements.txt
COPY main.py /app/main.py
COPY gunicorn_config.py /app/gunicorn_config.py
COPY models /app/models
COPY www /app/www

EXPOSE 5000
WORKDIR /app
ENTRYPOINT ["gunicorn", "--config", "gunicorn_config.py", "main:app"]