FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# change software source
COPY ./docker/sources.list /etc/apt/sources.list
COPY ./docker/.condarc /root/.condarc
COPY ./docker/pip.conf /root/.pip/pip.conf


# apt software
RUN apt-get update && apt-get install -y wget gnuplot git vim pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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
