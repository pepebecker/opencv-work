# OpenCV Work

## Build and enter into the Docker Container
```shell
make
make enter
```

## Rotate and Crop the Video
```shell
cd src
./video.py --rotate 90
```

This will take the 90CC.mp4 from the `videos` directory and rotate it by 90 degree, then it will crop the video to a resolution of 1024x768 and save it to the `output` directory.