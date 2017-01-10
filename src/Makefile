video:
	./video.py -i ../videos/90CC.mp4 -r 90 -o ../output/90CC.mp4

stream:
	./stream.py -i http://$(DOCKER_HOST_IP):8090/test.mjpeg -r 90 -k 4 -o ../output/90CC.mp4

frames:
	@for i in `seq -f "%02g" 1 12`; do \
		./frame.py -i ../portraits/$$i.jpg -o ../output/$$i; \
		echo Saved to ../output/$$i.jpg; \
	done
