# OpenCV Work

## Build and enter into the Docker Container
```shell
make
make enter
```

### To use the command line tool please navigate to the `src` directory.

```shell
cd src
```

## Run the Example using the Makefile
```shell
make video
```

### Or run the command directly in your Terminal

```shell
./video.py -i ../videos/90CC.mp4 -r 90 -k 0 -o ../output/90CC.mp4
```

This will take the `90CC.mp4` from the `videos` directory and rotate it by 90 degree, then it will crop the video to a default resolution of 1024x768 and save it to the `output` directory.  
The `-k` option with the `0` paremeter tells the program to not skip any frames.

## For a Complete List of Options run this Command
```shell
./video.py --help
```