FROM python:3.12.7-slim-bookworm AS base
LABEL maintainer="Shenghao Wang <shenghao.wsh@gmail.com>"

WORKDIR /root

RUN pip install --upgrade pip
RUN apt update

##################################################### DEPS-NO-PIN ENVIRONMENT #####################################################
FROM base AS deps-no-pin
COPY requirements-no-pin.txt .
RUN pip install --no-cache-dir -r requirements-no-pin.txt

##################################################### DEPENDENCIES SETUP ##########################################################
FROM base AS final

# Update apt get
RUN apt-get update --fix-missing

# Install linux packages required by repo
RUN apt -y install make git

# Remove apt binaries
RUN rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/root:./src"

# COPY the actual code
COPY . /root
