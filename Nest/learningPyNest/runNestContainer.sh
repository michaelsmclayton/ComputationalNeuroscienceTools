
############################################
# Get docker image
############################################
# docker pull neuralensemble/simulation

############################################
# Run container (with local files mounted)
############################################
IMAGE_NAME=neuralensemble/simulation
CONTAINER_NAME=neuro_simulator
# NEST_MODELS_DR=/home/docker/packages/nest-2.14.0/models
docker rm $CONTAINER_NAME
SRC_DR=$(pwd) # State source directory
docker run \
    --name $CONTAINER_NAME \
    --volume "$SRC_DR/code:/code" \
    --workdir=/code \
    -it $IMAGE_NAME /bin/bash


# -----------------------------------------------------------------
# A Docker image for running neuronal network simulations
# -----------------------------------------------------------------

# FROM neuralensemble/base
# MAINTAINER andrew.davison@unic.cnrs-gif.fr

# ENV NEST_VER=2.14.0 NRN_VER=7.4
# ENV NEST=nest-$NEST_VER NRN=nrn-$NRN_VER
# ENV PATH=$PATH:$VENV/bin
# RUN ln -s /usr/bin/2to3-3.4 $VENV/bin/2to3

# WORKDIR $HOME/packages
# RUN wget https://github.com/nest/nest-simulator/releases/download/v$NEST_VER/nest-$NEST_VER.tar.gz -O $HOME/packages/$NEST.tar.gz;
# RUN wget http://www.neuron.yale.edu/ftp/neuron/versions/v$NRN_VER/$NRN.tar.gz
# RUN tar xzf $NEST.tar.gz; tar xzf $NRN.tar.gz; rm $NEST.tar.gz $NRN.tar.gz
# RUN git clone --depth 1 https://github.com/INCF/libneurosim.git
# RUN cd libneurosim; ./autogen.sh

# RUN mkdir $VENV/build
# WORKDIR $VENV/build
# RUN mkdir libneurosim; \
#     cd libneurosim; \
#     PYTHON=$VENV/bin/python $HOME/packages/libneurosim/configure --prefix=$VENV; \
#     make; make install; ls $VENV/lib $VENV/include
# RUN mkdir $NEST; \
#     cd $NEST; \
#     ln -s /usr/lib/python3.4/config-3.4m-x86_64-linux-gnu/libpython3.4.so $VENV/lib/; \
#     cmake -DCMAKE_INSTALL_PREFIX=$VENV \
#           -Dwith-mpi=ON  \
#           ###-Dwith-music=ON \
#           -Dwith-libneurosim=ON \
#           -DPYTHON_LIBRARY=$VENV/lib/libpython3.4.so \
#           -DPYTHON_INCLUDE_DIR=/usr/include/python3.4 \
#           $HOME/packages/$NEST; \
#     make; make install
# RUN mkdir $NRN; \
#     cd $NRN; \
#     $HOME/packages/$NRN/configure --with-paranrn --with-nrnpython=$VENV/bin/python --disable-rx3d --without-iv --prefix=$VENV; \
#     make; make install; \
#     cd src/nrnpython; $VENV/bin/python setup.py install; \
#     cd $VENV/bin; ln -s ../x86_64/bin/nrnivmodl

# RUN $VENV/bin/pip3 install lazyarray nrnutils PyNN
# RUN $VENV/bin/pip3 install brian2

# WORKDIR /home/docker/
# RUN echo "source $VENV/bin/activate" >> .bashrc


# -----------------------------------------------------------------
# A base Docker image for Python-based computational neuroscience and neurophysiology
# -----------------------------------------------------------------
#
# FROM neurodebian:jessie
# MAINTAINER andrew.davison@unic.cnrs-gif.fr

# ENV DEBIAN_FRONTEND noninteractive
# ENV LANG=C.UTF-8

# RUN apt-get update; apt-get install -y automake libtool build-essential openmpi-bin libopenmpi-dev git vim  \
#                        wget python3 libpython3-dev libncurses5-dev libreadline-dev libgsl0-dev cython3 \
#                        python3-pip python3-numpy python3-scipy python3-matplotlib python3-jinja2 python3-mock \
#                        ipython3 python3-httplib2 python3-docutils python3-yaml \
#                        subversion python3-venv python3-mpi4py python3-tables python3-h5py cmake

# RUN useradd -ms /bin/bash docker
# USER docker
# ENV HOME=/home/docker
# RUN mkdir $HOME/env; mkdir $HOME/packages

# ENV VENV=$HOME/env/neurosci

# # we run venv twice because of this bug: https://bugs.python.org/issue24875
# # using the workaround proposed by Georges Racinet
# RUN python3 -m venv $VENV && python3 -m venv --system-site-packages $VENV

# RUN $VENV/bin/pip3 install --upgrade pip
# RUN $VENV/bin/pip3 install parameters quantities neo "django<1.9" django-tagging future hgapi gitpython sumatra nixio
# RUN $VENV/bin/pip3 install --upgrade nose ipython