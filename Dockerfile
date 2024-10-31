FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y zsh
RUN pip install poetry==1.8.4
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

WORKDIR /home
COPY pyproject.toml .

# Configure poetry to create the virtualenv inside the project directory
RUN poetry config virtualenvs.in-project true
RUN poetry install

# Use the virtual environment's Python directly
CMD [ "zsh" ]