FROM ubuntu:22.04
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y python3 python3-pip

RUN pip install streamlit llama-index-legacy setuptools && \
    mkdir /src && \
    mkdir /.kb

COPY /src /src

EXPOSE 8501

CMD ["streamlit", "run", "/src/chataro_bot.py"]
