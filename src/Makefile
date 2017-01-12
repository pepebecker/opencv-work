video:
	mkdir -p ../output
	ffmpeg -loglevel panic -i ../videos/90CC.mp4 -f image2pipe - | ./process.py -r 90 | ffmpeg -r 30 -i - -an -vcodec libx264 ../output/video.mp4

video2frames:
	ffmpeg -loglevel panic -i ../videos/90CC.mp4 -f image2pipe - | ./process.py -r 90 | ./jpeg.py -o ../output -n video2frame_%04d.jpg

frames:
	./frames.py -i ../frames | ./process.py -r 90 | ./jpeg.py -o ../output -n frame_%04d.jpg

frames2video:
	mkdir -p ../output
	./frames.py -i ../frames | ./process.py -r 90 | ffmpeg -r 15 -i - -an -vcodec libx264 ../output/frames2video.mp4

landmarks:
	@for i in `seq -f "%02g" 1 12`; do \
		./landmarks.py -i ../portraits/$$i.jpg -o ../output/$$i; \
		echo Saved to ../output/$$i.jpg; \
	done

stream:
	./stream.py -i http://$(DOCKER_HOST_IP):8090/test.mjpeg -r 90 -k 4 -o ../output/90CC.mp4