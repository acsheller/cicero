# SUBER Docker

If you are reading this, you are in the docker folder of the SUBER for MIND dataset project.
The development environment has been containerized for simplicity.

## Building the container

Ideally, you would want it to be your UID and GID so do this:

```.docker
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t subercon .

```

## Running in VSCode


## Run with docker run
