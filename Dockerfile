FROM wusuopu/python-opencv3-dlib:py3.5

RUN pip install --upgrade pip
RUN pip install -U scikit-image
RUN pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose

RUN mkdir -p /usr/src/app

ADD . /usr/src/app

WORKDIR /usr/src/app
