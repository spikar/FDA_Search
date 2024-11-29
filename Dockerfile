FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt requirements.txt 

RUN pip3 install -r requirements.txt

COPY app.py  /app
COPY prompts.py /app
COPY .env /app
COPY configs.py /app
COPY core.py /app
COPY get_retriever.py /app
COPY multiquery.py /app
ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]
