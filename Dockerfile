FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# change software source
COPY ./docker/sources.list /etc/apt/sources.list
COPY ./docker/.condarc /root/.condarc
COPY ./docker/pip.conf /root/.pip/pip.conf


# apt software
RUN apt-get update && \
    apt-get install -y wget gnuplot git vim python3-pip g++-7 gcc-7 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# miniconda
WORKDIR /app
COPY . /app/
ENV PATH="/miniconda3/bin:$PATH"
# installation
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda-latest.sh \
    && bash ./miniconda-latest.sh -b -p /miniconda3 \
    && ln -s /miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc \
    && find /miniconda3/ -follow -type f -name '*.a' -delete \
    && find /miniconda3/ -follow -type f -name '*.js.map' -delete \
    && conda clean -afy
# create environment

RUN conda create --name sgnn --file /app/docker/spec-list.txt \
    && conda clean -afy \
    && echo "conda activate sgnn" >> ~/.bashrc

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-n", "sgnn", "/bin/bash", "-c"]

WORKDIR /app
RUN wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1-1ubuntu2.1~18.04.23_amd64.deb \ 
    && dpkg -i libssl1.1_1.1.1-1ubuntu2.1~18.04.23_amd64.deb 

RUN apt-get update && \
    apt-get install -y libssl-dev

RUN wget https://cmake.org/files/v3.27/cmake-3.27.7.tar.gz \
    && tar -xzvf cmake-3.27.7.tar.gz \
    && cd cmake-3.27.7 \
    && ./bootstrap --prefix=/usr/local \
    && make -j8 \
    && make install \
    && echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bashrc \
    && source ~/.bashrc \
    && ln -s /usr/local/bin/cmake /usr/bin/cmake


RUN git clone --recurse-submodules https://github.com/BearBiscuit05/signn_dgl_0.9.git 

WORKDIR /app/signn_dgl_0.9    

RUN chmod a+x ./rebuild.sh \
    && bash ./rebuild.sh
