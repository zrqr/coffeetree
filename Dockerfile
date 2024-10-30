FROM python:3.12.7-bookworm

RUN pip install poetry==1.8.4
WORKDIR /home
COPY . .
CMD ["/bin/bash"]