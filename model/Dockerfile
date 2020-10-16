FROM python:3.7

RUN pip install fastapi uvicorn numpy

EXPOSE 8070


COPY . /app

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8070"]
