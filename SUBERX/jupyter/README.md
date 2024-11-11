# SUBERX -- new SUBER just for MIND

To build the docker container

```.bash

docker build -t suber-image -f Dockerfile.suber .

```

To run it

```.bash

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -p 8889:8889 -v ./SUBERX:/app/SUBERX --name suber-container suber-image

```
