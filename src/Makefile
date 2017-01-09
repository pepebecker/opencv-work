video:
	./video.py -i ../videos/90CC.mp4 -r 90 -o ../output/90CC.mp4

stream:
	./stream.py -i ../videos/90CC.mp4 -r 90 -k 0 -o ../output/90CC.mp4

frames:
	@for i in `seq -f "%02g" 1 12`; do \
		./frame.py -i ../portraits/$$i.jpg -o ../output/$$i; \
		echo Saved to ../output/$$i.jpg; \
	done
