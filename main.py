from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn
import pickle
import json

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = pickle.load(open('ai-challenge.pkl', 'rb'))
ohe = pickle.load(open('ohe.pkl', 'rb'))
scaler = pickle.load(open('scl.pkl', 'rb'))


class Model_input(BaseModel):
    category: str
    type: str
    year: int
    month: str


@app.get('/')
def index():
    return {'message': 'Hello, World'}


def transform(cat1, cat2, num, cat3):
    category_input = [cat1, cat2, cat3]
    x = ohe.transform([category_input])
    y = scaler.transform([[num]])
    z = model.predict(np.concatenate((y, x), axis=1))
    return z[0]


# print(transform('Alkoholunfälle', 'insgesamt', 2021, '01'))


@app.post('/predict')
def prediction(data: Model_input):
    input_data = data.json()
    input_dictionary = json.loads(input_data)
    category = input_dictionary['category']
    type = input_dictionary['type']
    year = input_dictionary['year']
    month = input_dictionary['month']
    prediction = transform(category, type, year, month)
    return {'prediction': prediction.tolist()}


# print(prediction(data))
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
