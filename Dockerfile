# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
FROM jupyter/scipy-notebook

# Get some karma ##
MAINTAINER Florian Briegel, briegel@mpia.de

LABEL org.opencontainers.image.source = "https://github.com/sdss/lvmjupyter"

RUN pip3 install ipydatagrid
RUN pip3 install jupyter-app-launcher

RUN pip install jupyterlab_widgets

USER root

RUN mkdir /usr/local/bin/start-notebook.d
USER ${NB_UID}

ENTRYPOINT start-notebook.sh --LabApp.token=''
