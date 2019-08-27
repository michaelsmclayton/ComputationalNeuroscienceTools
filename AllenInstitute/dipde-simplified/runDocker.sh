
# Build docker image
docker build --tag alleninstitute/dipde:ubuntu16 -f ./docker/ubuntu_16.04_python.dockerfile .

# Run Docker image
export DIPDEDIR=`pwd`
docker run -v $DIPDEDIR:/python/dipde -t -it alleninstitute/dipde:ubuntu16 /bin/bash
# python setup.py test