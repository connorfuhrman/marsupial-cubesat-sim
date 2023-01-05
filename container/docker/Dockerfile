####################################################################################################
# Dockerfile for DSRC Project
#
# This Dockerfile creates a container image holding the DSRC Python code
####################################################################################################


FROM python:3.9

RUN apt update && apt upgrade -y && pip install --upgrade pip --no-cache

# Copy the repo
COPY . /app/

# Install the repo
WORKDIR /app
RUN pip install . --no-cache


CMD ["echo", "Invoke a script with python /path/from/toplevel/dir"]
