FROM debian

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential python3 python3-dev curl git python3-pip python3 openjdk-11-jre
ADD . .

RUN pip3 install -r eval-requirements.txt

CMD ["python3", "-i", "eval_chapterization.py"]

