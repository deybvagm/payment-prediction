from django.shortcuts import render
from django.http import HttpResponse
import os
from django.conf import settings
# Create your views here.

# Code for loading the model
from sklearn.externals import joblib
import pandas as pd
import numpy as np

print(settings.BASE_DIR)
model_path = os.path.join(settings.BASE_DIR, "credits", "static","credits", "lgb.pkl")
print(model_path)
model = joblib.load('credits/static/credits/lgb.pkl')
# users_idxs = [244379, 284293]
# train = pd.read_csv('credits/static/credits/train.csv')
# test = pd.read_csv('credits/static/credits/test.csv')
users = pd.read_csv('credits/static/credits/predict_data.csv')
print('users shape ', users.values.shape)
# users = train.iloc[users_idxs, :]

# users = users[['CNT_CHILDREN', 'AMT_CREDIT', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'bureau_CREDIT_DAY_OVERDUE_count']]
users_partial_info = users[['CNT_CHILDREN', 'AMT_CREDIT', 'DAYS_BIRTH', 'DAYS_EMPLOYED']]
users_partial_info['DAYS_BIRTH'] = users_partial_info['DAYS_BIRTH'].abs()/365
users_partial_info['DAYS_EMPLOYED'] = users_partial_info['DAYS_EMPLOYED'].abs()/365
cols = ['HIJOS', 'CREDITO', 'EDAD', 'TIEMPO_EN_EMPRESA']
# cols = ['HIJOS', 'CREDITO', 'EDAD', 'TIEMPO_EN_EMPRESA', 'ATRASOS_CREDITOS']
users_partial_info.columns = cols

# END code for loading model

# def home(request):
#     return HttpResponse('baba')

def home(request):
    return render(request, 'credits/home.html', {'data': users_partial_info.values.astype(int).tolist()})
    # return HttpResponse(users.to_html())


def prediction(request):
    out = model.predict(users.values)
    # print(out)
    pred_data = np.zeros((users_partial_info.values.shape[0], users_partial_info.values.shape[1] + 1))
    pred_data[:, :pred_data.shape[1] -1] = users_partial_info.values
    pred_data[:, -1] = out

    print(pred_data.shape)
    # print('la predicción para el primer usuario 1 es: ', 'VA A TENER UN MAL COMPORTAMIENTO EN LOS PAGOS' if out[0]==1 else 'VA A TENER UN BUEN COMPORTAMIENTO EN LOS PAGOS')
    return render(request, 'credits/prediction.html', {'preds': out, 'data': pred_data.astype(int).tolist()})
