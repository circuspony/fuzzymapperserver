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
import uuid
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import statsmodels.api as sm
from sklearn.decomposition import PCA
from fuzzy_clustering.core.fuzzy import GK


app = Flask(__name__)
CORS(app) 

@app.route('/')
def index():
    return 'Index Page'

def findOutliersIn(dataArray):
    newObjects = []
    for valueArray in dataArray:
        valueArrayNew = list(map(lambda v: float(v), valueArray))
        valueArrayNew = np.array(valueArrayNew)
        valueArrayNew= np.sort(valueArrayNew) 
        q25=np.quantile(valueArrayNew, 0.25)
        q75=np.quantile(valueArrayNew, 0.75)
        IQD=q75-q25
        outlierLower15 = q25-1.5*IQD
        outlierHigher15 = q75+1.5*IQD
        outlierLower30 = q25-3*IQD
        outlierHigher30 = q75+3*IQD
        objectSet={}
        objectSet["ol15"]=outlierLower15
        objectSet["oh15"]=outlierHigher15
        objectSet["ol30"]=outlierLower30
        objectSet["oh30"]=outlierHigher30
        outlierArray =[]
        membershipsArray =[]
        for v in valueArray:
            on = outlierNumber(v,outlierLower15,outlierHigher15)
            outlierArray.append(on)
            if on==1:
                membershipsArray.append(outlierLowMembership(float(v),float(outlierLower15),float(outlierLower30)))
            if on==2:
                membershipsArray.append(outlierHighMembership(float(v),float(outlierHigher15),float(outlierHigher30)))
            if on==0:
                membershipsArray.append(0)
        objectSet["outlierArray"]=outlierArray
        objectSet["membershipsArray"]=membershipsArray
        newObjects.append(objectSet)
    outlierArray = newObjects[0]["outlierArray"]
    membershipsArray = newObjects[0]["membershipsArray"]
    for a in newObjects:
        for oi in range(len(a["outlierArray"])):
            if a["outlierArray"][oi]!=0:
                outlierArray[oi]=a["outlierArray"][oi]
            if a["membershipsArray"][oi]!=0:
                membershipsArray[oi]=a["membershipsArray"][oi]
    return {
        "o":outlierArray,
        "m":membershipsArray
    }

@app.route('/clusters', methods=['POST'])
async def clusters():
    if request.method == 'POST':
        content = request.json
        outliers = findOutliersIn(content["data"])
        l0 = map(lambda x: [x], content["data"][0])
        result =  np.array(list(l0))
        for index in range(len(content["data"])):
            if (index!=0):
                l0 = map(lambda x: [x], content["data"][index])
                a0 =  np.array(list(l0))
                result = np.hstack((result,a0))
        X = result.astype(float)
        newX =  []
        for notOi in range(len(outliers["o"])):
            if outliers["o"][notOi]==0:
                newX.append(X[notOi].tolist())
        newX =  np.array(newX)
        if content["outlier"]==True:
            X=newX
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
        if content["outlier"]==True:
            for i in range(len(fcm_labels)):
                label=fcm_labels[i]
                label.insert(0, 0)
                label.append(0)
                fcm_labels[i]=label
            for notOi in range(len(outliers["o"])):
                if outliers["o"][notOi]==1:
                    m = [0]*len(fcm_labels[0])
                    m[0]=outliers["m"][notOi]
                    fcm_labels.insert(notOi, m)
                if outliers["o"][notOi]==2:
                    m = [0]*len(fcm_labels[0])
                    m[len(fcm_labels[0])-1]=outliers["m"][notOi]
                    fcm_labels.insert(notOi, m)
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
        return {
        "status": "ok",
        "corr":df.corr().to_numpy().tolist(),
        "headers": {"Access-Control-Allow-Origin": "*"}
        }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }

def is_float(element: any) -> bool:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

def is_int(element: any) -> bool:
    if element is None: 
        return False
    try:
        int(element)
        return True
    except ValueError:
        return False

def outlierNumber(num,lb,hb):
    if float(num)>hb:
        return 2
    if float(num)<lb:
        return 1
    return 0

def outlierFuzzy(data):
    newObjects = []
    for objectSet in data:
        isNumber = True
        for o in objectSet["values"]:
            if is_int(o["value"])==False and is_float(o["value"])==False:
                isNumber = False
                break
        if (isNumber):
            objectArray = list(map(lambda os: float(os["value"]), objectSet["values"]))
            objectArray = np.array(objectArray)
            objectArray= np.sort(objectArray) 
            q25=np.quantile(objectArray, 0.25)
            q75=np.quantile(objectArray, 0.75)
            IQD=q75-q25
            outlierLower15 = q25-1.5*IQD
            outlierHigher15 = q75+1.5*IQD
            outlierLower30 = q25-3*IQD
            outlierHigher30 = q75+3*IQD
            objectSet["ol15"]=outlierLower15
            objectSet["oh15"]=outlierHigher15
            objectSet["ol30"]=outlierLower30
            objectSet["oh30"]=outlierHigher30
            for o in objectSet["values"]:
                o["outlier"]=outlierNumber(o["value"],outlierLower15,outlierHigher15)
            newObjects.append(objectSet)
    return newObjects

@app.route('/outlier', methods=['POST'])
async def outlier():
    if request.method == 'POST':
        newObjects = []
        for objectSet in request.json["data"]:
            outliers = outlierFuzzy(objectSet)
            newObjects.append(outliers)
        return {
        "status": "ok",
        "objectSet":newObjects,
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

def outlierLowMembership(val,o15,o30):
    if val<o30:
        return 1
    membershipValue = (o15-val)/(o15-o30)
    return membershipValue

def outlierHighMembership(val,o15,o30):
    if val>o30:
        return 1
    membershipValue = (val-o15)/(o30-o15)
    return membershipValue    

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
                    if fs["outlier"] == True:
                        if value["outlier"]==0:
                            mv.insert(0, 0)
                            mv.append(0)
                        if value["outlier"]==1:
                            ml = outlierLowMembership(float(value["value"]),float(fs["ol15"]),float(fs["ol30"]))
                            mv.insert(0, ml)
                            mv.append(0)
                        if value["outlier"]==2:
                            mv.insert(0, 0)
                            mh=outlierHighMembership(float(value["value"]),float(fs["oh15"]),float(fs["oh30"]))
                            mv.append(mh)
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
                    if fs["outlier"] == True:
                        if value["outlier"]==0:
                            mv.insert(0, 0)
                            mv.append(0)
                        if value["outlier"]==1:
                            ml = outlierLowMembership(float(value["value"]),float(fs["ol15"]),float(fs["ol30"]))
                            mv.insert(0, ml)
                            mv.append(0)
                        if value["outlier"]==2:
                            mv.insert(0, 0)
                            mh=outlierHighMembership(float(value["value"]),float(fs["oh15"]),float(fs["oh30"]))
                            mv.append(mh)
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

def outlierRegression(x,y):

    X= np.sort(x) 
    q25=np.quantile(X, 0.25)
    q75=np.quantile(X, 0.75)
    IQD=q75-q25
    outlierLower15 = q25-1.5*IQD
    outlierHigher15 = q75+1.5*IQD
    outlierLower30 = q25-3*IQD
    outlierHigher30 = q75+3*IQD
    outliers = []
    for xo in x:
        if xo>outlierHigher15 or xo<outlierLower15:
            outliers.append(1)
            continue
        outliers.append(0)
    Y= np.sort(y) 
    q25=np.quantile(Y, 0.25)
    q75=np.quantile(Y, 0.75)
    IQD=q75-q25
    outlierLower15 = q25-1.5*IQD
    outlierHigher15 = q75+1.5*IQD
    outlierLower30 = q25-3*IQD
    outlierHigher30 = q75+3*IQD
    for yi in range(len(y)):
        if y[yi]>outlierHigher15 or y[yi]<outlierLower15:
            outliers[yi]=1
    return outliers
    
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
                outliers = outlierRegression(X,y)
                newX=[]
                newY=[]
                for xi in range(len(X)):
                    if outliers[xi]==0:
                        newX.append(X[xi])
                        newY.append(y[xi])
                if (content["data"]["outlier"]==True):
                    X=np.array(newX)
                    y=np.array(newY)
                X_ = sm.add_constant(X.reshape(-1,1))
                model = sm.OLS(y, X_).fit()
                predictions = model.predict(X_)
                b = model.params.tolist()[1]
                a = model.params.tolist()[0]
                r2 = model.rsquared
                F = model.fvalue
                p = model.f_pvalue
                data={
                    "from":fsi["name"],
                    "to":fei["name"],
                    "b":b,
                    "r2":r2,
                    "F":F,
                    "p":p,
                    "x":X.tolist(),
                    "y":y.tolist(),
                    "a":a
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
        
        outliers = outlierRegression(X,y)
        newX=[]
        newY=[]
        for xi in range(len(X)):
            if outliers[xi]==0:
                newX.append(X[xi])
                newY.append(y[xi])
        if (content["data"]["outlier"]==True):
            X=np.array(newX)
            y=np.array(newY)
        X_ = sm.add_constant(X.reshape(-1,1))
        model = sm.OLS(y, X_).fit()
        predictions = model.predict(X_)
        b = model.params.tolist()[1]
        a = model.params.tolist()[0]
        r2 = model.rsquared
        F = model.fvalue
        p = model.f_pvalue
        data = {
                    "b":b,
                    "r2":r2,
                    "F":F,
                    "p":p,
                    "x":X.tolist(),
                    "y":y.tolist(),
                    "a":a
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
            if summr == 0:
                result.append(0)
                continue

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

@app.route('/accumulation', methods=['POST'])
async def accumulation():
    if request.method == 'POST':
        content = request.json
        newObjects = []
        for objectIndex in range(len(content["data"][0]["fevals"])):
            upper=0
            lower=0
            rulemu = []
            ruley = []
            for combosArray in content["icombos"]:
                for rule in content["rulebase"]:
                    mu = 1
                    ysum=0
                    for termIndex in range(len(rule["combo"])):
                        factor=content["data"][termIndex]
                        
                        factorLabels = list(map(lambda l: str(l), factor["labels"]))
                        termLabelIndex=factorLabels.index(str(rule["combo"][termIndex]))
                        objectMu = factor["fevals"][objectIndex][termLabelIndex]
                        if (mu>objectMu):
                            mu = objectMu
                    for termIndex in range(len(rule["evals"])):
                        ysum=ysum+rule["evals"][termIndex]/sum(rule["evals"])*float(combosArray[termIndex]["values"][objectIndex])
                    rulemu.append(mu) 
                    ruley.append(ysum)
            for index in range(len(rulemu)):
                upper=upper+rulemu[index]*ruley[index]
                lower=lower+rulemu[index]
            result = 0
            if lower!=0:
                result=upper/lower
            newObjects.append(result)
        return {
        "status": "ok",
        "predictions":newObjects,
        "headers": {"Access-Control-Allow-Origin": "*"}
        }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }

@app.route('/test', methods=['POST'])
async def test():
    if request.method == 'POST':
        return {
        "status": "ok",
        "headers": {"Access-Control-Allow-Origin": "*"}
        }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }
