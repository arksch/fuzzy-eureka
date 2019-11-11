FROM jupyter/scipy-notebook:1386e2046833

USER $NB_UID

RUN conda install --quiet --yes gudhi

RUN pip install --user \
    pybind11==2.3.0 \
    numpy==1.17.2 \
    scipy==1.3.1 \
    cechmate==0.0.8 \
    intervaltree==3.0.2 \
    persim==0.1.0 \
    ipdb==0.11 \
    altair==3.2.0 \
    pytest==4.2.0 \
    pytest-cov==2.8.1


# Install Perseus
RUN mkdir work/Perseus && \
    wget -O work/Perseus/perseus "https://people.maths.ox.ac.uk/nanda/source/perseusLin" && \
    chmod +x /home/jovyan/work/Perseus/perseus
ENV PERSEUSPATH /home/jovyan/work/Perseus/perseus

# Clone the git repo and checkout the commit the experiment is based on
RUN git clone https://github.com/arksch/fuzzy-eureka work/fuzzy-eureka && \
    cd work/fuzzy-eureka && \
    git checkout ae69db5b871321771def2caa69934ff7095ddf7f && \
    pip install --user -e .
