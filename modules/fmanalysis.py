import numpy as np
import pandas as pd


def fmtest():
    print("test test")


def outlierLowMembership(val, o15, o30):
    if val < o30:
        return 1
    membershipValue = (o15-val)/(o15-o30)
    return membershipValue


def outlierHighMembership(val, o15, o30):
    if val > o30:
        return 1
    membershipValue = (val-o15)/(o30-o15)
    return membershipValue


def outlierRegression(x, y):
    X = np.sort(x)
    q25 = np.quantile(X, 0.25)
    q75 = np.quantile(X, 0.75)
    IQD = q75-q25
    outlierLower15 = q25-1.5*IQD
    outlierHigher15 = q75+1.5*IQD
    outlierLower30 = q25-3*IQD
    outlierHigher30 = q75+3*IQD
    outliers = []
    for xo in x:
        if xo > outlierHigher15 or xo < outlierLower15:
            outliers.append(1)
            continue
        outliers.append(0)
    Y = np.sort(y)
    q25 = np.quantile(Y, 0.25)
    q75 = np.quantile(Y, 0.75)
    IQD = q75-q25
    outlierLower15 = q25-1.5*IQD
    outlierHigher15 = q75+1.5*IQD
    outlierLower30 = q25-3*IQD
    outlierHigher30 = q75+3*IQD
    for yi in range(len(y)):
        if y[yi] > outlierHigher15 or y[yi] < outlierLower15:
            outliers[yi] = 1
    return outliers


def findOutliersIn(dataArray):
    newObjects = []
    for valueArray in dataArray:
        valueArrayNew = list(map(lambda v: float(v), valueArray))
        valueArrayNew = np.array(valueArrayNew)
        valueArrayNew = np.sort(valueArrayNew)
        q25 = np.quantile(valueArrayNew, 0.25)
        q75 = np.quantile(valueArrayNew, 0.75)
        IQD = q75-q25
        outlierLower15 = q25-1.5*IQD
        outlierHigher15 = q75+1.5*IQD
        outlierLower30 = q25-3*IQD
        outlierHigher30 = q75+3*IQD
        objectSet = {}
        objectSet["ol15"] = outlierLower15
        objectSet["oh15"] = outlierHigher15
        objectSet["ol30"] = outlierLower30
        objectSet["oh30"] = outlierHigher30
        outlierArray = []
        membershipsArray = []
        for v in valueArray:
            on = outlierNumber(v, outlierLower15, outlierHigher15)
            outlierArray.append(on)
            if on == 1:
                membershipsArray.append(outlierLowMembership(
                    float(v), float(outlierLower15), float(outlierLower30)))
            if on == 2:
                membershipsArray.append(outlierHighMembership(
                    float(v), float(outlierHigher15), float(outlierHigher30)))
            if on == 0:
                membershipsArray.append(0)
        objectSet["outlierArray"] = outlierArray
        objectSet["membershipsArray"] = membershipsArray
        newObjects.append(objectSet)
    outlierArray = newObjects[0]["outlierArray"]
    membershipsArray = newObjects[0]["membershipsArray"]
    for a in newObjects:
        for oi in range(len(a["outlierArray"])):
            if a["outlierArray"][oi] != 0:
                outlierArray[oi] = a["outlierArray"][oi]
            if a["membershipsArray"][oi] != 0:
                membershipsArray[oi] = a["membershipsArray"][oi]
    return {
        "o": outlierArray,
        "m": membershipsArray
    }


def outlierNumber(num, lb, hb):
    if float(num) > hb:
        return 2
    if float(num) < lb:
        return 1
    return 0


def outlierFuzzy(data):
    newObjects = []
    for objectSet in data:
        isNumber = True
        for o in objectSet["values"]:
            if is_int(o["value"]) == False and is_float(o["value"]) == False:
                isNumber = False
                break
        if (isNumber):
            objectArray = list(
                map(lambda os: float(os["value"]), objectSet["values"]))
            objectArray = np.array(objectArray)
            objectArray = np.sort(objectArray)
            q25 = np.quantile(objectArray, 0.25)
            q75 = np.quantile(objectArray, 0.75)
            IQD = q75-q25
            outlierLower15 = q25-1.5*IQD
            outlierHigher15 = q75+1.5*IQD
            outlierLower30 = q25-3*IQD
            outlierHigher30 = q75+3*IQD
            objectSet["ol15"] = outlierLower15
            objectSet["oh15"] = outlierHigher15
            objectSet["ol30"] = outlierLower30
            objectSet["oh30"] = outlierHigher30
            for o in objectSet["values"]:
                o["outlier"] = outlierNumber(
                    o["value"], outlierLower15, outlierHigher15)
            newObjects.append(objectSet)
    return newObjects


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


def makeDynamic(data):
    newData = []
    names = []
    keyNames = []
    dateName = None
    fdict = {}
    for objectSet in data:
        if (objectSet["key"]):
            keyNames.append(objectSet["title"])
        else:
            if (objectSet["date"]):
                dateName = objectSet["title"]
            else:
                names.append(objectSet["title"])
        v = list(map(lambda v: v["value"], objectSet["values"]))
        fdict[objectSet["title"]] = v
    df = pd.DataFrame(fdict)
    uniqueNames = df[keyNames[0]].unique()
    dfNew = pd.DataFrame({})
    for name in uniqueNames:
        dfTemp = pd.DataFrame({})
        dfTemp = df[df[keyNames[0]] == name].reset_index(drop=True)
        dfTemp[dateName] = dfTemp[dateName].astype(float)
        minDate = dfTemp[dateName].min()
        minDateIndex = dfTemp[dfTemp[dateName] == minDate].index.item()
        for restName in names:
            dfTemp[restName] = (dfTemp[restName]).astype(float)
            dfTemp[restName] = dfTemp[restName] - \
                float(dfTemp[restName][minDateIndex])
        dfNew = pd.concat([dfNew, dfTemp], ignore_index=True)
    for restName in names:
        dfNew[restName] = np.where(
            dfNew[restName] < 0, dfNew[restName]/abs(dfNew[restName].min()), dfNew[restName])
        dfNew[restName] = np.where(
            dfNew[restName] > 0, dfNew[restName]/dfNew[restName].max(), dfNew[restName])
    for dataset in data:
        newVals = dfNew[dataset["title"]].values.tolist()
        newValuesFull = []
        for dpi in range(len(dataset["values"])):
            val = dataset["values"][dpi]
            val["value"] = newVals[dpi]
            newValuesFull.append(val)
        dataset["values"] = newValuesFull
        newData.append(dataset)
    return newData


# async def correlations():
#     if request.method == 'POST':
#         content = request.json
#         dfData = {}
#         for nameIndex in range(len(content["fields"])):
#             dfData[content["fields"][nameIndex]] = list(
#                 np.float_(content["data"][nameIndex]))
#         pd.set_option('display.max_colwidth', 0)
#         df = pd.DataFrame(data=dfData)
#         return {
#             "status": "ok",
#             "corr": df.corr().to_numpy().tolist(),
#             "headers": {"Access-Control-Allow-Origin": "*"}
#         }
#     return {
#         "status": "error",
#         "headers": {"Access-Control-Allow-Origin": "*"}
#     }
