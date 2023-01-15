
SHELL := /bin/bash

IMAGE = dockerfilespython

THIS_FILE_PATH := $(realpath $(lastword $(MAKEFILE_LIST)))
THIS_FILE_DIR := $(shell dirname $(THIS_FILE_PATH))

train: check-model
	python3 training.py -m "$(MODEL)" 


build: check-tag
	DOCKER_BUILDKIT=1 docker build --pull --rm \
		-f "Dockerfile.distroless3.9" \
		-t $(IMAGE):$(TAG) \
		--build-arg "DISTRO=$(DISTRO)" \
		$(THIS_FILE_DIR)

run: check-tag
	docker run -ti --rm \
		-p 8000:8000 \
		$(IMAGE):$(TAG) mockup.py

check-model:
ifndef MODEL 
	$(error MODEL is not set.) 
endif

check-tag:
ifeq ($(TAG),distroless)
DISTRO="gcr.io/distroless/python3"
else ifeq ($(TAG),baseline)
DISTRO="python:3.9.15-bullseye"
else ifeq ($(TAG),slim)
DISTRO="python:3.9.15-slim-bullseye"
else
	$(error DISTRO is not set.) 
endif

help:
	$(info train: `make train MODEL=LogisiticRegression`.) 
	$(info build: `make build TAG=baseline`.) 
	$(info run: `make run TAG=baseline`.) 
