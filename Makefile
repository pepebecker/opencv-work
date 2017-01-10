LOCAL_PATH = $(shell pwd)
DOCKER_PATH = /usr/src/app

IMAGE = face-tracking

build:
	docker build -t $(IMAGE) .

enter:
	@docker run -v /dev/null:/dev/raw1394 --rm -it -p 3000:3000 -v $(LOCAL_PATH):$(DOCKER_PATH) $(IMAGE) $(shell ipconfig getifaddr en0)

serve:
	ffserver -f ffserver.conf
