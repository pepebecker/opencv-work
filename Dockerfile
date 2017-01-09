FROM wusuopu/python-opencv3-dlib:py3.5

RUN apt-get -y update
RUN apt-get -y install mesa-utils xvfb libgl1-mesa-dri libglapi-mesa libosmesa6
RUN apt-get -y install freeglut3

RUN pip install --upgrade pip
RUN pip install --upgrade --user scikit-image
RUN pip install --upgrade --user numpy scipy matplotlib ipython jupyter pandas sympy nose
RUN pip install --upgrade --user PyOpenGL PyOpenGL_accelerate

RUN mkdir -p /usr/src/app

ADD . /usr/src/app

WORKDIR /usr/src/app/src

# X Virtual Frame Buffer
ENV DISPLAY :99
# ADD run.sh /run.sh
# RUN chmod a+x /run.sh
# CMD /run.sh
CMD /usr/src/app/run.sh
