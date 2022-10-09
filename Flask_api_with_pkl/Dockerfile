FROM python:3.9.7

WORKDIR /code
#You need to run this command for get packages from env that you work: pip list --format=freeze > requirements.txt
COPY ./requirements.txt /code/requirements.txt 

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY predict.py /code/predict.py

COPY diabet_model.pkl /code/diabet_model.pkl

CMD ["python" , "/code/predict.py"]

#Image build command: docker build --tag diabete_image .
#Container running command: docker run --name diabete_predict_container --restart=on-failure:10 -d -p 9696:9696 diabete_image

