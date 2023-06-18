from modules.fmanalysis import fmtest
from modules.fmanalysis import findOutliersIn
from modules.fmanalysis import outlierRegression
from modules.fmanalysis import outlierFuzzy
from modules.fmanalysis import makeDynamic
from modules.fmanalysis import outlierLowMembership
from modules.fmanalysis import outlierHighMembership


from modules.fmfactors import calculateIndirectY
from modules.fmfactors import triangleMembership
from modules.fmfactors import trapezoidMembership
from modules.fmfactors import fss


import itertools
from sklearn.preprocessing import scale, normalize
from sklearn.preprocessing import StandardScaler
from fuzzy_clustering.core.fuzzy import GK
from sklearn.decomposition import PCA
import statsmodels.api as sm
from flask import Flask
from flask import request
from flask import send_from_directory
from flask_cors import CORS
import numpy as np
from fcmeans import FCM
import pandas as pd
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return 'Index Page'


@app.route('/clusters', methods=['POST'])
async def clusters():
    if request.method == 'POST':
        content = request.json
        outliers = findOutliersIn(content["data"])
        l0 = map(lambda x: [x], content["data"][0])
        result = np.array(list(l0))
        for index in range(len(content["data"])):
            if (index != 0):
                l0 = map(lambda x: [x], content["data"][index])
                a0 = np.array(list(l0))
                result = np.hstack((result, a0))

        X = result.astype(float)
        newX = []
        for notOi in range(len(outliers["o"])):
            if outliers["o"][notOi] == 0:
                newX.append(X[notOi].tolist())
        newX = np.array(newX)
        if content["outlier"] == True:
            X = newX

        scaler = StandardScaler()

        if content["standard"] == True:
            X = scaler.fit_transform(X)
        fcm_centers = None
        fcm_labels = None
        fuzzyS = 0
        if content["cmethod"] == "CM":
            fcm = FCM(n_clusters=content["clusters"])
            fcm.fit(X)
            fcm_centers = fcm.centers.tolist()
            fcm_labels = fcm.soft_predict(X).tolist()
            fuzzyS = fss(X, fcm.soft_predict(X), content["clusters"])
        if content["cmethod"] == "GK":
            m, v, f = GK(X, c=content["clusters"])
            fcm_centers = v.tolist()
            fcm_labels = m.tolist()
            fuzzyS = fss(X, m, content["clusters"])
        if content["outlier"] == True:
            for i in range(len(fcm_labels)):
                label = fcm_labels[i]
                label.insert(0, 0)
                label.append(0)
                fcm_labels[i] = label
            for notOi in range(len(outliers["o"])):
                if outliers["o"][notOi] == 1:
                    m = [0]*len(fcm_labels[0])
                    m[0] = outliers["m"][notOi]
                    fcm_labels.insert(notOi, m)
                if outliers["o"][notOi] == 2:
                    m = [0]*len(fcm_labels[0])
                    m[len(fcm_labels[0])-1] = outliers["m"][notOi]
                    fcm_labels.insert(notOi, m)
        return {
            "status": "ok",
            "centers": fcm_centers,
            "labels": fcm_labels,
            "fuzzyS": fuzzyS,
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
            dfData[content["fields"][nameIndex]] = list(
                np.float_(content["data"][nameIndex]))
        pd.set_option('display.max_colwidth', 0)
        df = pd.DataFrame(data=dfData)
        return {
            "status": "ok",
            "corr": df.corr().to_numpy().tolist(),
            "headers": {"Access-Control-Allow-Origin": "*"}
        }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }


@app.route('/outlier', methods=['POST'])
async def outlier():
    if request.method == 'POST':
        data = []
        newObjects = []
        if request.json["dynamic"] == True:
            for objectSet in request.json["data"]:
                data.append(makeDynamic(objectSet))
        else:
            data = request.json["data"]
        for objectSet in data:
            outliers = outlierFuzzy(objectSet)
            newObjects.append(outliers)
        return {
            "status": "ok",
            "objectSet": newObjects,
            "headers": {"Access-Control-Allow-Origin": "*"}
        }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }


def findByKey(key, currentEvals):
    for ce in range(len(currentEvals)):
        if currentEvals[ce]["key"] == key:
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
                    mv = triangleMembership(value["value"], fs["functionSets"])
                    if fs["outlier"] == True:
                        if value["outlier"] == 0:
                            mv.insert(0, 0)
                            mv.append(0)
                        if value["outlier"] == 1:
                            ml = outlierLowMembership(
                                float(value["value"]), float(fs["ol15"]), float(fs["ol30"]))
                            mv.insert(0, ml)
                            mv.append(0)
                        if value["outlier"] == 2:
                            mv.insert(0, 0)
                            mh = outlierHighMembership(
                                float(value["value"]), float(fs["oh15"]), float(fs["oh30"]))
                            mv.append(mh)
                    if fs["external"] == True:
                        mv = list(map(lambda v: min(v, value["eval"]), mv))
                    mv = list(
                        map(lambda v: min(v, abs(float(fs["coef"]))), mv))
                    if float(fs["coef"]) < 0:
                        mv = mv[::-1]
                    newValue = value.copy()
                    newValue["mv"] = mv
                    newValues.append(newValue)
            if fs["type"] == 1:
                for value in fs["values"]:
                    mv = trapezoidMembership(
                        value["value"], fs["functionSets"])
                    if fs["outlier"] == True:
                        if value["outlier"] == 0:
                            mv.insert(0, 0)
                            mv.append(0)
                        if value["outlier"] == 1:
                            ml = outlierLowMembership(
                                float(value["value"]), float(fs["ol15"]), float(fs["ol30"]))
                            mv.insert(0, ml)
                            mv.append(0)
                        if value["outlier"] == 2:
                            mv.insert(0, 0)
                            mh = outlierHighMembership(
                                float(value["value"]), float(fs["oh15"]), float(fs["oh30"]))
                            mv.append(mh)
                    if fs["external"] == True:
                        mv = list(map(lambda v: min(v, value["eval"]), mv))
                    mv = list(
                        map(lambda v: min(v, abs(float(fs["coef"]))), mv))
                    if float(fs["coef"]) < 0:
                        mv = mv[::-1]
                    newValue = value.copy()
                    newValue["mv"] = mv
                    newValues.append(newValue)
            newDict = {}
            newDict["values"] = newValues
            newDict["name"] = fs["name"]
            newDict["title"] = fs["title"]
            # result.append(newDict)
            result.append(newValues)
        currentEvals = result[0].copy()
        for r in result[1:]:
            for v in r:
                index = findByKey(v["key"], currentEvals)
                if (index >= 0):
                    ceCopy = currentEvals[index].copy()
                    mv = list(map(lambda x, y: max(
                        x, y), v["mv"], ceCopy["mv"]))
                    # check outliers for combos
                    if ceCopy["outlier"] > 0:
                        mv = ceCopy["mv"]
                    if v["outlier"] > 0:
                        mv = v["mv"]
                    ceCopy["mv"] = mv
                    currentEvals[index] = ceCopy
                else:
                    currentEvals.append(v)
        currentEvals = sorted(currentEvals, key=lambda d: d['key'])
        return {
            "status": "ok",
            "evals": currentEvals,
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
        newRegressions = []

        for fsi in content["data"]["fsi"]:
            for fei in content["data"]["fei"]:
                X = np.array(list(fsi["values"])).astype(float)
                y = np.array(list(fei["values"])).astype(float)
                clusterLabels = []

                if (content["data"]["iv"] == None):
                    cl = map(lambda x: [1], X)
                    clusterLabels = list(cl)
                else:
                    clusterLabels = content["data"]["iv"]
                l = map(lambda x: x.index(max(x)), clusterLabels)
                clusterLabelsNonFuzzy = np.array(list(l))
                outliers = outlierRegression(X, y)
                newX = []
                newY = []
                clusterLabelsNonFuzzyNew = []
                for xi in range(len(X)):
                    if outliers[xi] == 0:
                        newX.append(X[xi])
                        newY.append(y[xi])
                        clusterLabelsNonFuzzyNew.append(
                            clusterLabelsNonFuzzy[xi])
                if (content["data"]["outlier"] == True):
                    X = np.array(newX)
                    y = np.array(newY)
                    clusterLabelsNonFuzzy = clusterLabelsNonFuzzyNew
                legacyX = X
                legacyY = y
                pairInfo = []
                for clusterIndex in range(len(clusterLabels[0])):
                    X = legacyX
                    y = legacyY
                    newX = []
                    newY = []
                    for xi in range(len(clusterLabelsNonFuzzy)):
                        if clusterLabelsNonFuzzy[xi] == clusterIndex:
                            newX.append(X[xi])
                            newY.append(y[xi])
                    X = newX
                    y = newY
                    if len(X) <= 1 or len(y) <= 1:
                        data = {
                            "from": fsi["name"],
                            "to": fei["name"],
                            "term": clusterIndex,
                            "b": 0,
                            "r2": 0,
                            "F": 0,
                            "p": 0,
                            "x": [],
                            "y": [],
                            "a": 0
                        }
                        pairInfo.append(data)
                        continue
                    X = (X-np.min(X))/(np.max(X)-np.min(X))
                    y = (y-np.min(y))/(np.max(y)-np.min(y))
                    X_ = sm.add_constant(X.reshape(-1, 1))
                    model = sm.OLS(y, X_).fit()
                    b = model.params.tolist()[1]
                    a = model.params.tolist()[0]
                    r2 = model.rsquared
                    F = model.fvalue
                    p = model.f_pvalue
                    data = {
                        "from": fsi["name"],
                        "to": fei["name"],
                        "term": clusterIndex,
                        "b": b,
                        "r2": r2,
                        "F": F,
                        "p": p,
                        "x": X.tolist(),
                        "y": y.tolist(),
                        "a": a
                    }
                    pairInfo.append(data)
                newRegressions.append(pairInfo)

        regressions = []
        for fsi in content["data"]["fsi"]:
            for fei in content["data"]["fei"]:
                X = np.array(list(fsi["values"])).astype(float)
                y = np.array(list(fei["values"])).astype(float)
                outliers = outlierRegression(X, y)
                newX = []
                newY = []
                for xi in range(len(X)):
                    if outliers[xi] == 0:
                        newX.append(X[xi])
                        newY.append(y[xi])
                if (content["data"]["outlier"] == True):
                    X = np.array(newX)
                    y = np.array(newY)
                X = (X-np.min(X))/(np.max(X)-np.min(X))
                y = (y-np.min(y))/(np.max(y)-np.min(y))
                X_ = sm.add_constant(X.reshape(-1, 1))
                model = sm.OLS(y, X_).fit()
                predictions = model.predict(X_)
                b = model.params.tolist()[1]
                a = model.params.tolist()[0]
                r2 = model.rsquared
                F = model.fvalue
                p = model.f_pvalue
                data = {
                    "from": fsi["name"],
                    "to": fei["name"],
                    "b": b,
                    "r2": r2,
                    "F": F,
                    "p": p,
                    "x": X.tolist(),
                    "y": y.tolist(),
                    "a": a
                }
                regressions.append(data)

        return {
            "status": "ok",
            "regressions": regressions,
            "newRegressions": newRegressions,
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

        newRegressions = []
        x = content["data"]["fsi"]
        y = content["data"]["fei"]
        xr = np.array(x[0]).astype(float)
        yr = np.array(y[0]).astype(float)

        scaler = StandardScaler()
        # xr = xr.reshape(-1,1)
        # xr = scaler.fit_transform(xr)
        # xr = list(map(lambda x: x[0], xr))
        # yr = yr.reshape(-1,1)
        # yr = scaler.fit_transform(yr)
        # yr = list(map(lambda x: x[0], yr))

        # xr =  (xr-np.min(xr))/(np.max(xr)-np.min(xr))
        # yr =  (yr-np.min(yr))/(np.max(yr)-np.min(yr))

        pca = PCA(n_components=1, whiten=True)
        if len(x) > 1:
            for xri in x[1:]:
                xra = np.array(xri).astype(float)

                # xra = xra.reshape(-1,1)
                # xra = scaler.fit_transform(xra)
                # xra = list(map(lambda x: x[0], xra))

                # xra =  (xra-np.min(xra))/(np.max(xra)-np.min(xra))
                xr = np.column_stack((xr, xra))
            xr1 = scaler.fit_transform(xr)
            xr = pca.fit_transform(xr1)
        else:
            l = map(lambda x: [x], xr)
            xr = np.array(list(l))
        if len(y) > 1:
            for yri in y[1:]:
                yra = np.array(yri).astype(float)
                # yra = yra.reshape(-1,1)
                # yra = scaler.fit_transform(yra)
                # yra = list(map(lambda x: x[0], yra))

                # yra =  (yra-np.min(yra))/(np.max(yra)-np.min(yra))
                yr = np.column_stack((yr, yra))
            yr1 = scaler.fit_transform(yr)
            yr = pca.fit_transform(yr1)
        else:
            l = map(lambda x: [x], yr)
            yr = np.array(list(l))
        if content["data"]["reverseX"] == True:
            l = map(lambda x: [-x[0]], xr)
            xr = np.array(list(l))
        if content["data"]["reverseY"] == True:
            l = map(lambda y: [-y[0]], yr)
            yr = np.array(list(l))
        # l = map(lambda x: x[0], xr)
        # xr =  np.array(list(l))
        # l = map(lambda x: x[0], yr)
        # yr =  np.array(list(l))
        X = xr
        y = yr

        outliers = outlierRegression(X, y)
        clusterLabels = []
        if (content["data"]["iv"] == None):
            cl = map(lambda x: [1], X)
            clusterLabels = list(cl)
        else:
            clusterLabels = content["data"]["iv"]
        l = map(lambda x: x.index(max(x)), clusterLabels)
        clusterLabelsNonFuzzy = np.array(list(l))
        newX = []
        newY = []
        clusterLabelsNonFuzzyNew = []

        for xi in range(len(X)):
            if outliers[xi] == 0:
                newX.append(X[xi])
                newY.append(y[xi])
                clusterLabelsNonFuzzyNew.append(clusterLabelsNonFuzzy[xi])
        if (content["data"]["outlier"] == True):
            X = np.array(newX)
            y = np.array(newY)
            clusterLabelsNonFuzzy = clusterLabelsNonFuzzyNew

        legacyX = X
        legacyY = y
        pairInfo = []
        for clusterIndex in range(len(clusterLabels[0])):
            X = legacyX
            y = legacyY
            newX = []
            newY = []
            for xi in range(len(clusterLabelsNonFuzzy)):
                if clusterLabelsNonFuzzy[xi] == clusterIndex:
                    newX.append(X[xi])
                    newY.append(y[xi])
            X = newX
            y = newY
            if len(X) <= 1 or len(y) <= 1:
                data = {
                    "term": clusterIndex,
                    "b": 0,
                    "r2": 0,
                    "F": 0,
                    "p": 0,
                    "x": [],
                    "y": [],
                    "a": 0
                }
                pairInfo.append(data)
                continue
            X = (X-np.min(X))/(np.max(X)-np.min(X))
            y = (y-np.min(y))/(np.max(y)-np.min(y))
            X_ = sm.add_constant(X.reshape(-1, 1))
            model = sm.OLS(y, X_).fit()
            b = model.params.tolist()[1]
            a = model.params.tolist()[0]
            r2 = model.rsquared
            F = model.fvalue
            p = model.f_pvalue
            data = {
                "term": clusterIndex,
                "b": b,
                "r2": r2,
                "F": F,
                "p": p,
                "x": X.tolist(),
                "y": y.tolist(),
                "a": a
            }
            pairInfo.append(data)

        X = legacyX
        y = legacyY

        X = (X-np.min(X))/(np.max(X)-np.min(X))
        y = (y-np.min(y))/(np.max(y)-np.min(y))
        X_ = sm.add_constant(X.reshape(-1, 1))
        model = sm.OLS(y, X_).fit()
        predictions = model.predict(X_)
        b = model.params.tolist()[1]
        a = model.params.tolist()[0]
        r2 = model.rsquared
        F = model.fvalue
        p = model.f_pvalue
        data = {
            "b": b,
            "r2": r2,
            "F": F,
            "p": p,
            "x": X.tolist(),
            "y": y.tolist(),
            "a": a
        }
        return {
            "status": "ok",
            "regression": data,
            "newRegression": pairInfo,
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
        ic = len(labels1[0])
        oc = len(labels2[0])
        matrix = [[0]*len(labels2[0]) for i in range(len(labels1[0]))]
        l = map(lambda x: x.index(max(x)), labels1)
        labels1 = np.array(list(l))
        l = map(lambda x: x.index(max(x)), labels2)
        labels2 = np.array(list(l))
        for li in range(len(labels1)):
            matrix[labels1[li]][labels2[li]] += 1
        result = []
        for mr in range(len(matrix)):
            summr = sum(matrix[mr])
            if summr == 0:
                result.append(0)
                continue

            maxPercent = -999
            for m in range(len(matrix[mr])):
                if matrix[mr][m]/summr > maxPercent:
                    maxPercent = matrix[mr][m]/summr
                matrix[mr][m] = matrix[mr][m]/summr
            v = (maxPercent-1/oc)/(1-1/oc)
            result.append(v)
        return {
            "status": "ok",
            "result": result,
            "matrix": matrix,
            "headers": {"Access-Control-Allow-Origin": "*"}
        }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }


@app.route('/indirectacc', methods=['POST'])
async def indirectacc():
    if request.method == 'POST':
        content = request.json
        factors = content["factors"]
        newFactors = []
        # prepare
        for f in factors:
            data = {}
            if (f["evals"] == None):
                cl = map(lambda x: [1], f["indicators"][0]["values"])
                f["evals"] = list(cl)

            if len(f["influence"]) != len(f["evals"][0]):
                data["influence"] = [f["influence"][0]]*len(f["evals"][0])
            else:
                data["influence"] = f["influence"]
            print(data)
            findicators = []
            for fi in f["indicators"]:
                evalValues = []
                data["coef"] = fi["coef"]
                for evalIndex in range(len(f["evals"][0])):
                    nonZeroELements = []
                    nonZeroELementsValues = []
                    for v in range(len(fi["values"])):
                        if f["evals"][v][evalIndex] > 0:
                            nonZeroELements.append(
                                {"obj": v, "eval": evalIndex, "evalValue": f["evals"][v][evalIndex], "coef": fi["coef"]})
                            nonZeroELementsValues.append(fi["values"][v])

                    if len(nonZeroELementsValues) > 1:
                        nonZeroELementsValues = np.array(
                            nonZeroELementsValues).astype(float)
                        nonZeroELementsValues = (nonZeroELementsValues-np.min(nonZeroELementsValues))/(
                            np.max(nonZeroELementsValues)-np.min(nonZeroELementsValues))
                    if len(nonZeroELementsValues) == 1:
                        nonZeroELementsValues[0] = 1
                    for nzei in range(len(nonZeroELementsValues)):
                        nonZeroELements[nzei]["value"] = nonZeroELementsValues[nzei]
                    evalValues.append(nonZeroELements)
                findicators.append(evalValues)
            data["findicators"] = findicators
            newFactors.append(data)
        # go through objects to get predictions
        predictions = []
        for objectIndex in range(len(factors[0]["indicators"][0]["values"])):
            xs = []
            for f in newFactors:
                myX = []
                for fi in f["findicators"]:
                    for el in fi:
                        for eli in el:
                            if eli["obj"] == objectIndex:
                                myX.append(eli)
                xs.append(myX)
            Y = calculateIndirectY(xs, newFactors)
            predictions.append(Y)
        return {
            "status": "ok",
            "predictions": predictions,
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
            upper = 0
            lower = 0
            rulemu = []
            ruley = []
            for combosArray in content["icombos"]:
                for rule in content["rulebase"]:
                    mu = 1
                    ysum = 0
                    for termIndex in range(len(rule["combo"])):
                        factor = content["data"][termIndex]

                        factorLabels = list(
                            map(lambda l: str(l), factor["labels"]))
                        termLabelIndex = factorLabels.index(
                            str(rule["combo"][termIndex]))
                        objectMu = factor["fevals"][objectIndex][termLabelIndex]
                        if (mu > objectMu):
                            mu = objectMu
                    for termIndex in range(len(rule["evals"])):
                        ysum = ysum+rule["evals"][termIndex]/sum(rule["evals"])*float(
                            combosArray[termIndex]["values"][objectIndex])
                    rulemu.append(mu)
                    ruley.append(ysum)
            for index in range(len(rulemu)):
                upper = upper+rulemu[index]*ruley[index]
                lower = lower+rulemu[index]
            result = 0
            if lower != 0:
                result = upper/lower
            newObjects.append(result)
        return {
            "status": "ok",
            "predictions": newObjects,
            "headers": {"Access-Control-Allow-Origin": "*"}
        }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }


@app.route('/pcaprognosis', methods=['POST'])
async def pcaprognosis():
    if request.method == 'POST':
        content = request.json["data"]
        currentObj = content["currentPrognosisObject"]
        indChanges = content["indIn"]
        iv = content["iv"]
        l = map(lambda x: x.index(max(x)), iv)
        clusterLabelsNonFuzzy = np.array(list(l))
        neededCluster = clusterLabelsNonFuzzy[currentObj]
        connection = content["connection"]
        fcEval = []
        if len(iv[0]) == len(connection["fcEval"]):
            fcEval = connection["fcEval"]
        else:
            fcEval = [connection["fcEval"][0]]*len(iv[0])

        newValues = []

        x = content["fsi"]
        y = content["fei"]
        xr = np.array(x[0]["values"]).astype(float)
        newValueX = float(x[0]["values"][currentObj])+float(x[0]
                                                            ["change"])*0.01*float(x[0]["values"][currentObj])
        xr = np.append(xr, newValueX)
        newValues.append(newValueX)
        yr = np.array(y[0]["values"]).astype(float)
        scaler = StandardScaler()
        pca = PCA(n_components=1, whiten=True)
        if len(x) > 1:
            for xri in x[1:]:
                xra = np.array(xri["values"]).astype(float)
                newValueX = float(
                    xri["values"][currentObj])+float(xri["change"])*0.01*float(xri["values"][currentObj])
                newValues.append(newValueX)
                xra = np.append(xra, newValueX)
                xr = np.column_stack((xr, xra))
            xr1 = scaler.fit_transform(xr)
            xr = pca.fit_transform(xr1)
        else:
            l = map(lambda x: [x], xr)
            xr = np.array(list(l))
        if len(y) > 1:
            for yri in y[1:]:
                yra = np.array(yri["values"]).astype(float)
                yr = np.column_stack((yr, yra))
            yr1 = scaler.fit_transform(yr)
            yr = pca.fit_transform(yr1)
        else:
            l = map(lambda x: [x], yr)
            yr = np.array(list(l))
        X = xr
        y = yr
        newCountX = X[-1][0]
        oldCountX = X[currentObj][0]

        newX = []
        newY = []
        currentObjNew = 0
        for xi in range(len(clusterLabelsNonFuzzy)):
            if xi == currentObj:
                currentObjNew = len(newX)
            if clusterLabelsNonFuzzy[xi] == neededCluster:
                newX.append(X[xi][0])
                newY.append(y[xi][0])

        maxX = np.max(X)
        minX = np.min(X)
        maxY = np.max(y)
        minY = np.min(y)

        X = (X-np.min(X))/(np.max(X)-np.min(X))
        y = (y-np.min(y))/(np.max(y)-np.min(y))

        newCountX = (newCountX-minX)/(maxX-minX)
        oldCountX = X[currentObjNew][0]
        oldCountY = y[currentObjNew][0]
        bPCA = fcEval[neededCluster]
        newCountY = oldCountY+(newCountX-oldCountX)*bPCA

        return {
            "status": "ok",
            "newValues": newValues,
            "oldCountY": oldCountY,
            "newCountY": newCountY,
            "headers": {"Access-Control-Allow-Origin": "*"}
        }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }


@app.route('/test', methods=['POST'])
async def test():
    if request.method == 'POST':
        fmtest()
        return {
            "status": "ok",
            "headers": {"Access-Control-Allow-Origin": "*"}
        }
    return {
        "status": "error",
        "headers": {"Access-Control-Allow-Origin": "*"}
    }
