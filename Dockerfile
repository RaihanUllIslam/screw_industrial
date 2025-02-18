FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get clean
RUN apt-get update && apt-get install -y git tmux libopencv-dev libgl1-mesa-dev apt-utils wget unzip
RUN pip3 install --upgrade pip

COPY requirements.txt /workdir/requirements.txt

WORKDIR /workdir

RUN pip install -r requirements.txt
CMD ["bash"]
