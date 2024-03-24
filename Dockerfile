FROM ubuntu:22.04
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y python3 python3-pip

RUN mkdir /src && \
    mkdir /data && \
    mkdir /storage && \
    mkdir /updata

COPY /src /src
COPY requirements.txt /root/

RUN pip install -r /root/requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "/src/chataro_bot.py"]
