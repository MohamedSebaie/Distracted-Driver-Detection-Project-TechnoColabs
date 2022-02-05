FROM python:3.7
ADD . /Distracted_Driver_Detection_DockerV5
WORKDIR /Distracted_Driver_Detection_DockerV5
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y