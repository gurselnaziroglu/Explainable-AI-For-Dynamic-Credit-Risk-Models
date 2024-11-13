# syntax=docker/dockerfile:1


ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app
ARG UID=10001


# Get compilers + make
RUN apt-get update && apt-get install gcc -y && apt-get install g++ -y && apt-get install make

# Set python path
ENV PYTHONPATH "${PYTHONPATH}:/app/"

# create a virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN . /venv/bin/activate 

# install dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt


# Fix timeshap
RUN head -n -2 ../venv/lib/python3.10/site-packages/timeshap/plot/__init__.py > temp_file && mv temp_file ../venv/lib/python3.10/site-packages/timeshap/plot/__init__.py
RUN head -n -2 ../venv/lib/python3.10/site-packages/timeshap/explainer/__init__.py > temp_file && mv temp_file ../venv/lib/python3.10/site-packages/timeshap/explainer/__init__.py


# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE 8501

# Just a check for the port
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health


# Run the application.
CMD make dashboard --always-make
