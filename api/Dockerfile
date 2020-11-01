FROM python:3.7

COPY requirements.txt requirements.txt


RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .


EXPOSE 8050


CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8050"]
