from flask import Flask
from flask import request
from flask import send_from_directory
from flask_cors import CORS 
import numpy as np
from fcmeans import FCM
import pandas as pd
import dataframe_image as dfi
import calendar
import time
import math
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import statsmodels.api as sm
from sklearn.decomposition import PCA
from fuzzy_clustering.core.fuzzy import GK


app = Flask(__name__)
CORS(app) 

@app.route('/')
def index():
    return 'Index Page'

@app.route('/clusters', methods=['POST'])
async def clusters():
    if request.method == 'POST':
        content = request.json
        l0 = map(lambda x: [x], content["data"][0])
        result =  np.array(list(l0))
        for index in range(len(content["data"])):
            if (index!=0):
                l0 = map(lambda x: [x], content["data"][index])
                a0 =  np.array(list(l0))
                result = np.hstack((result,a0))
        X = result.astype(float)
        print(X)
        fcm_centers=None
        fcm_labels=None
        if content["cmethod"]=="CM":
            fcm = FCM(n_clusters=content["clusters"])
            fcm.fit(X)
            fcm_centers = fcm.centers.tolist()
            fcm_labels = fcm.soft_predict(X).tolist()
        if content["cmethod"]=="GK":
          m,v,f = GK(X,c = content["clusters"])
          fcm_centers = v.tolist()
          fcm_labels = m.tolist()
        return {
        "status": "ok",
        "centers":fcm_centers,
        "labels":fcm_labels,
        "headers": {"Access-Control-Allow-Origin": "*"}
    }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }

@app.route('/correlations', methods=['POST'])
async def correlations():
    if request.method == 'POST':
        content = request.json
        dfData = {}
        for nameIndex in range(len(content["fields"])):
            dfData[content["fields"][nameIndex]]= list(np.float_(content["data"][nameIndex])) 
        pd.set_option('display.max_colwidth', 0)
        df = pd.DataFrame(data=dfData)
        corr = df.corr().style.set_table_styles([{'selector': 'th', 'props': 'background-color: rgb(245 243 255)'}]).set_table_styles([{'selector': 'tr', 'props': 'background-color: rgb(245 243 255)'}]).background_gradient(cmap='coolwarm').set_precision(3)
        fname = 'pictures/{}.png'.format(calendar.timegm(time.gmtime()))
        dfi.export(corr, fname)
        return {
        "status": "ok",
        "corr":fname,
        "headers": {"Access-Control-Allow-Origin": "*"}
        }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }

@app.route('/triangle', methods=['POST'])
def triangle():
    if request.method == 'POST':
        content = request.json
        fig = Figure(figsize=(7,5))
        axis = fig.add_subplot(1, 1, 1)
        for i in range(len(content["data"])):
            if i==0:
                x=content["data"][i]
                y=[1,1,0]
                axis.plot(x, y, c = np.random.rand(3,))
            if i==len(content["data"])-1:
                x=content["data"][i]
                y=[0,1,1]
                axis.plot(x, y, c = np.random.rand(3,))
            if i!=0 and i!=len(content["data"])-1:
                x=content["data"][i]
                y=[0,1,0]
                axis.plot(x, y, c = np.random.rand(3,))
        fig.gca().set_facecolor((0.960784,0.952941176,1))
        fig.patch.set_facecolor((0.960784,0.952941176,1))
        fname = 'pictures/{}.png'.format(uuid.uuid4())
        fig.savefig(fname)
        return {
        "status": "ok",
        "fname":fname,
        "headers": {"Access-Control-Allow-Origin": "*"}
        }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }

@app.route('/trapezoid', methods=['POST'])
def trapezoid():
    if request.method == 'POST':
        content = request.json
        fig = Figure(figsize=(7,5))
        axis = fig.add_subplot(1, 1, 1)
        for i in range(len(content["data"])):
            if i==0:
                x=content["data"][i]
                y=[1,1,1,0]
                axis.plot(x, y, c = np.random.rand(3,))
            if i==len(content["data"])-1:
                x=content["data"][i]
                y=[0,1,1,1]
                axis.plot(x, y, c = np.random.rand(3,))
            if i!=0 and i!=len(content["data"])-1:
                x=content["data"][i]
                y=[0,1,1,0]
                axis.plot(x, y, c = np.random.rand(3,))
        fig.gca().set_facecolor((0.960784,0.952941176,1))
        fig.patch.set_facecolor((0.960784,0.952941176,1))
        fname = 'pictures/{}.png'.format(uuid.uuid4())
        fig.savefig(fname)
        return {
        "status": "ok",
        "fname":fname,
        "headers": {"Access-Control-Allow-Origin": "*"}
        }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }

@app.route('/pictures/<path:path>')
def send_report(path):
    return send_from_directory('pictures', path)


def triangleMembership(value,functionSets):
    value = float(value)
    mv= []
    for fsi in range(len(functionSets)):
        membershipValue = 0
        if fsi == 0:
            if value<=functionSets[fsi][2] and value>=functionSets[fsi][1]:
                membershipValue = (functionSets[fsi][2]-value)/(functionSets[fsi][2]-functionSets[fsi][1])
        if fsi == len(functionSets)-1:
            if value<=functionSets[fsi][1] and value>=functionSets[fsi][0]:
                membershipValue = (value - functionSets[fsi][0])/(functionSets[fsi][1]-functionSets[fsi][0])
        if fsi != len(functionSets)-1 and fsi != 0:
            if value<=functionSets[fsi][2] and value>=functionSets[fsi][1]:
                membershipValue = (functionSets[fsi][2]-value)/(functionSets[fsi][2]-functionSets[fsi][1])
            if value<=functionSets[fsi][1] and value>=functionSets[fsi][0]:
                membershipValue = (value - functionSets[fsi][0])/(functionSets[fsi][1]-functionSets[fsi][0])
        mv.append(membershipValue)
    return mv

def trapezoidMembership(value,functionSets):
    value = float(value)
    mv= []
    for fsi in range(len(functionSets)):
        membershipValue = 0
        if fsi == 0:
            if value<=functionSets[fsi][2] and value>=functionSets[fsi][0]:
                membershipValue = 1
            if value<=functionSets[fsi][3] and value>=functionSets[fsi][2]:
                membershipValue = (functionSets[fsi][3]-value)/(functionSets[fsi][3]-functionSets[fsi][2])
        if fsi == len(functionSets)-1:
            if value<=functionSets[fsi][3] and value>=functionSets[fsi][1]:
                membershipValue = 1
            if value<=functionSets[fsi][1] and value>=functionSets[fsi][0]:
                membershipValue = (value - functionSets[fsi][0])/(functionSets[fsi][1]-functionSets[fsi][0])
        if fsi != len(functionSets)-1 and fsi != 0:
            if value<=functionSets[fsi][2] and value>=functionSets[fsi][1]:
                membershipValue = 1
            if value<=functionSets[fsi][3] and value>=functionSets[fsi][2]:
                membershipValue = (functionSets[fsi][3]-value)/(functionSets[fsi][3]-functionSets[fsi][2])
            if value<=functionSets[fsi][1] and value>=functionSets[fsi][0]:
                membershipValue = (value - functionSets[fsi][0])/(functionSets[fsi][1]-functionSets[fsi][0])
        mv.append(membershipValue)
    return mv

def findByKey(key,currentEvals):
    for ce in range(len(currentEvals)):
        if currentEvals[ce]["key"]==key:
            return ce
    return -1


@app.route('/fuzzy', methods=['POST'])
def fuzzy():
    if request.method == 'POST':
        content = request.json
        result = []
        for fs in content["data"]:
            newValues = []
            if fs["type"] == 0:
                for value in fs["values"]:
                    mv = triangleMembership(value["value"],fs["functionSets"])
                    if fs["external"] == True:
                        mv = list(map(lambda v: min(v,value["eval"]), mv))
                    mv = list(map(lambda v: min(v,abs(float(fs["coef"]))), mv))
                    if float(fs["coef"])<0:
                       mv= mv[::-1]
                    newValue = value.copy()
                    newValue["mv"] = mv
                    newValues.append(newValue)
            if fs["type"] == 1:
                for value in fs["values"]:
                    mv = trapezoidMembership(value["value"],fs["functionSets"])
                    if fs["external"] == True:
                        mv = list(map(lambda v: min(v,value["eval"]), mv))
                    mv = list(map(lambda v: min(v,abs(float(fs["coef"]))), mv))
                    if float(fs["coef"])<0:
                       mv= mv[::-1]
                    newValue = value.copy()
                    newValue["mv"] = mv
                    newValues.append(newValue)
            newDict = {}
            newDict["values"]=newValues
            newDict["name"]=fs["name"]
            newDict["title"]=fs["title"]
            # result.append(newDict)
            result.append(newValues)
        currentEvals = result[0].copy()
        for r in result[1:]:
            for v in r:
                index = findByKey(v["key"],currentEvals)
                if (index>=0):
                    ceCopy = currentEvals[index].copy()
                    mv = list(map(lambda x, y: max(x,y), v["mv"], ceCopy["mv"]))
                    ceCopy["mv"] = mv
                    currentEvals[index] = ceCopy
                else:
                    currentEvals.append(v)
        currentEvals = sorted(currentEvals, key=lambda d: d['key']) 
        return {
        "status": "ok",
        "evals":currentEvals,
        "headers": {"Access-Control-Allow-Origin": "*"}
        }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }

    
@app.route('/regression', methods=['POST'])
def regression():
    if request.method == 'POST':
        content = request.json
        regressions = []
        for fsi in content["data"]["fsi"]:
            for fei in content["data"]["fei"]:
                X = np.array(list(fsi["values"])).astype(float)
                X =  (X-np.min(X))/(np.max(X)-np.min(X))
                y = np.array(list(fei["values"])).astype(float)
                y =  (y-np.min(y))/(np.max(y)-np.min(y))
                X_ = sm.add_constant(X.reshape(-1,1))
                model = sm.OLS(y, X_).fit()
                predictions = model.predict(X_)
                fig = Figure(figsize=(7,5))
                axis = fig.add_subplot(1, 1, 1)
                axis.plot(X, predictions, c = np.random.rand(3,))
                axis.scatter(X, y)
                fig.gca().set_facecolor((0.960784,0.952941176,1))
                fig.patch.set_facecolor((0.960784,0.952941176,1))
                fname = 'pictures/{}.png'.format(uuid.uuid4())
                fig.savefig(fname)
                b = model.params.tolist()[1]
                r2 = model.rsquared
                F = model.fvalue
                p = model.f_pvalue
                data={
                    "from":fsi["name"],
                    "to":fei["name"],
                    "fname":fname,
                    "b":b,
                    "r2":r2,
                    "F":F,
                    "p":p
                }
                regressions.append(data)

        return {
        "status": "ok",
        "regressions":regressions,
        "headers": {"Access-Control-Allow-Origin": "*"}
        }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }

@app.route('/pca', methods=['POST'])
def pca():
    if request.method == 'POST':
        content = request.json
        x=content["data"]["fsi"]
        y=content["data"]["fei"]
        xr=np.array(x[0]).astype(float)
        yr=np.array(y[0]).astype(float)

        pca = PCA(n_components=1,whiten=True)
        if len(x)>1:
            for xri in x[1:]:
                xra=np.array(xri).astype(float)
                xr = np.column_stack((xr, xra))
            xr = pca.fit_transform(xr)
        else:
            l = map(lambda x: [x], xr)
            xr =  np.array(list(l))
        if len(y)>1:
            for yri in y[1:]:
                yra=np.array(yri).astype(float)
                yr = np.column_stack((yr, yra))
            yr = pca.fit_transform(yr)
        else:
            l = map(lambda x: [x], yr)
            yr =  np.array(list(l))
        if content["data"]["reverseX"]==True:
            l = map(lambda x: [-x[0]], xr)
            xr =  np.array(list(l))
        if content["data"]["reverseY"]==True:
            l = map(lambda y: [-y[0]], yr)
            yr =  np.array(list(l))

        # l = map(lambda x: x[0], xr)
        # xr =  np.array(list(l))
        # l = map(lambda x: x[0], yr)
        # yr =  np.array(list(l))
        X =  (xr-np.min(xr))/(np.max(xr)-np.min(xr))
        y =  (yr-np.min(yr))/(np.max(yr)-np.min(yr))
        X_ = sm.add_constant(X.reshape(-1,1))
        model = sm.OLS(y, X_).fit()
        predictions = model.predict(X_)
        fig = Figure(figsize=(7,5))
        axis = fig.add_subplot(1, 1, 1)
        axis.plot(X, predictions, c = np.random.rand(3,))
        axis.scatter(X, y)
        fig.gca().set_facecolor((0.960784,0.952941176,1))
        fig.patch.set_facecolor((0.960784,0.952941176,1))
        fname = 'pictures/{}.png'.format(uuid.uuid4())
        fig.savefig(fname)
        b = model.params.tolist()[1]
        r2 = model.rsquared
        F = model.fvalue
        p = model.f_pvalue
        data = {
                    "fname":fname,
                    "b":b,
                    "r2":r2,
                    "F":F,
                    "p":p
                }
        return {
        "status": "ok",
        "regression":data,
        "headers": {"Access-Control-Allow-Origin": "*"}
        }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }

    
@app.route('/clustercomp', methods=['POST'])
def clustercomp():
    if request.method == 'POST':
        content = request.json
        labels1 = content["data"]["iv"]
        labels2 = content["data"]["ov"]
        matrix = [[0]*len(labels2[0]) for i in range(len(labels1[0]))]
        l = map(lambda x: x.index(max(x)), labels1)
        labels1 =  np.array(list(l))
        l = map(lambda x: x.index(max(x)), labels2)
        labels2 =  np.array(list(l))
        for li in range(len(labels1)):
            matrix[labels1[li]][labels2[li]]+=1
        result = []
        for mr in range(len(matrix)):
            summr = sum(matrix[mr])
            maxPercent = -999
            for m in range(len(matrix[mr])):
                if matrix[mr][m]/summr>maxPercent:
                    maxPercent = matrix[mr][m]/summr
                matrix[mr][m]=matrix[mr][m]/summr
            v=(maxPercent-1/summr)/(1-1/summr)
            result.append(v)
        return {
        "status": "ok",
        "result":result,
        "matrix":matrix,
        "headers": {"Access-Control-Allow-Origin": "*"}
        }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }