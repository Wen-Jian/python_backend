# FROM python:3.7-slim as base
FROM python:3.6

# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1


# FROM base AS python-deps

# Install pipenv and compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev
RUN pip install pipenv

# Install python dependencies in /.venv
COPY . .
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --skip-lock --python 3.6


# FROM base AS runtime

# # Copy virtual env from python-deps stage
# COPY --from=python-deps /.venv /.venv
# ENV PATH="/.venv/bin:$PATH"

# # Create and switch to a new user
# RUN useradd --create-home appuser
# WORKDIR /home/appuser
# USER appuser

# # Install application into container
# COPY . .

# Run the application
CMD pipenv run python manage.py