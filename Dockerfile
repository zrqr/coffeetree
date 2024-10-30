FROM python:3.12.7-bookworm

RUN apt-get update && apt-get install -y zsh
RUN pip install poetry==1.8.4
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

WORKDIR /home
COPY pyproject.toml .
RUN poetry install

CMD [ "zsh", "-c", "poetry shell" ]