FROM pepebecker/python-opencv:latest

RUN mkdir -p /usr/src/app

ADD . /usr/src/app

WORKDIR /usr/src/app