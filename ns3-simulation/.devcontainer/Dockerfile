# Use existing image and add python
FROM notspecial/ns-3-dev

ENV TZ Europe/Zurich
ENV LANG C.UTF-8


# Add python for eval.
RUN apt-get update && apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.9 python3-pip python3-venv python3.9-distutils

RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install jupyter matplotlib pandas seaborn numpy scipy
