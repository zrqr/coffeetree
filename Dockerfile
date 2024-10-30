FROM python:3.12.7-bookworm

RUN apt-get update && apt-get install -y zsh
RUN pip install poetry==1.8.4
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN poetry env use python3.12
RUN poetry install
WORKDIR /home
ENTRYPOINT [ "zsh" ]
CMD ["poetry shell"]