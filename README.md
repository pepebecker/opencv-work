# OpenCV Work

## Build and enter into the Docker Container
```shell
make
make enter
```

## Run the Video Example using the Makefile
```shell
make video
```

This will take the `90CC.mp4` from the `videos` directory and rotate it by 90 degree, then it will crop the video to a default resolution of 1024x768 and save it to the `output` directory.  
The `-k` option with the `0` paremeter tells the program to not skip any frames.

## Run the Stream Example using the Makefile
```shell
make stream
```

This will load the video from a url `http://192.168.0.240:8090/test.mjpeg` and rotate it by 90 degree, then it will crop the video to a default resolution of 1024x768 and serve it on `localhost:3000` which you can open on your host machine.

## For a Complete List of Options run one of these Commands
```shell
./video.py --help
./stream.py --help
```