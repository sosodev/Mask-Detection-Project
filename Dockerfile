FROM python:3.6

WORKDIR /app

RUN wget -nc https://f000.backblazeb2.com/file/cs497-datasets/mask_rcnn_masked_faces.h5

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY server.py /app/server.py
COPY utils /app/utils
COPY templates /app/templates

ENV FLASK_APP=server.py

CMD python -m flask run --host=0.0.0.0 --port="$PORT"
