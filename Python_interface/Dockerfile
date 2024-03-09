# syntax=docker/dockerfile:1

# 1. Build stage
FROM continuumio/miniconda3 AS build

# Copied in to extract decode package version
COPY ./environment.yaml /environment.yaml
# Create conda environment
RUN conda config --set channel_priority strict && \
  conda update -n base -c defaults conda && \
  conda env create --name comet_env --file /environment.yaml

# Create standalone environment
RUN conda install conda-pack && \
  conda-pack -n comet_env -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && rm /tmp/env.tar && \
  /venv/bin/conda-unpack


# 2. Run stage
FROM debian AS run

# Copy python environment
COPY --from=build /venv /venv
# Copy training/fitting script
COPY . /scripts
# Directory to contain in/out files
RUN mkdir /files

ENV PATH="/venv/bin:$PATH"
WORKDIR /scripts
# Host must have nvidia-container-toolkit installed;
# add --gpus all to "docker run" command
# Example usage:
# docker run --gpus all -v mount_path:/data comet python -m docker_entrypoint --localizations_path /data/in/test_dataset_3D.csv --param_path /data/in/test_dataset_3D.yaml --output_path /data/out
CMD ["python", "-m", "docker_entrypoint"]
