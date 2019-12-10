FROM python:3.6

WORKDIR /app
ADD sources sources
ADD src src
ADD run.py run.py
ADD requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "run.py"]
