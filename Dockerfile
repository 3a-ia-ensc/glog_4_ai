FROM python:3.8

RUN python -m pip install --upgrade pip

COPY requirements.txt /
RUN pip install -r /requirements.txt
COPY models /app/models
COPY www /app/www

EXPOSE 5000
WORKDIR /app/www
CMD ["flask", "run", "--host=0.0.0.0"]

#FROM python:3.6-slim-stretch
#
#ADD requirements.txt /
#RUN pip install -r /requirements.txt
#
#ADD . /app
#WORKDIR /app
#
#EXPOSE 5000
#CMD [ "python" , "app.py"]