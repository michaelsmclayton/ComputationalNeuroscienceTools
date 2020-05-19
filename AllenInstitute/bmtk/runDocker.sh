# https://github.com/AllenInstitute/bmtk/tree/develop/docker

# Get docker image
# docker pull alleninstitute/bmtk

# Run interactive bmtk image
docker run -v $(pwd):/home/shared/workspace -t -it alleninstitute/bmtk python /home/shared/workspace/postDockerSetup.py