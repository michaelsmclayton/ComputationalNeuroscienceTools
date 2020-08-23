CONTAINER_NAME=graspy

# Build container
docker build -t $CONTAINER_NAME .

# Run container
docker rm $CONTAINER_NAME
SRC_DR=$(pwd) # State source directory
docker run \
    --name $CONTAINER_NAME \
    --volume "$SRC_DR:/code" \
    --workdir=/code \
    -it $CONTAINER_NAME /bin/bash
