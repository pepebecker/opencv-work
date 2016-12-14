LOCAL_PATH = $(shell pwd)
DOCKER_PATH = /usr/src/app

IMAGE = face-tracking

build:
	docker build -t $(IMAGE) .

enter:
	@docker run -v /dev/null:/dev/raw1394 --rm -it -v $(LOCAL_PATH):$(DOCKER_PATH) $(IMAGE)