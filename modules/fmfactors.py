from scipy.spatial import distance
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import itertools


def triangleMembership(value, functionSets):
    value = float(value)
    mv = []
    for fsi in range(len(functionSets)):
        membershipValue = 0
        if fsi == 0:
            if value <= functionSets[fsi][2] and value >= functionSets[fsi][1]:
                if (functionSets[fsi][2]-functionSets[fsi][1]) == 0:
                    membershipValue = 1
                else:
                    membershipValue = (
                        functionSets[fsi][2]-value)/(functionSets[fsi][2]-functionSets[fsi][1])
        if fsi == len(functionSets)-1:
            if value <= functionSets[fsi][1] and value >= functionSets[fsi][0]:
                if (functionSets[fsi][1]-functionSets[fsi][0]) == 0:
                    membershipValue = 1
                else:
                    membershipValue = (
                        value - functionSets[fsi][0])/(functionSets[fsi][1]-functionSets[fsi][0])
        if fsi != len(functionSets)-1 and fsi != 0:
            if value <= functionSets[fsi][2] and value >= functionSets[fsi][1]:
                if (functionSets[fsi][2]-functionSets[fsi][1]) == 0:
                    membershipValue = 1
                else:
                    membershipValue = (
                        functionSets[fsi][2]-value)/(functionSets[fsi][2]-functionSets[fsi][1])
            if value <= functionSets[fsi][1] and value >= functionSets[fsi][0]:
                if (functionSets[fsi][1]-functionSets[fsi][0]) == 0:
                    membershipValue = 1
                else:
                    membershipValue = (
                        value - functionSets[fsi][0])/(functionSets[fsi][1]-functionSets[fsi][0])
        mv.append(membershipValue)
    return mv


def trapezoidMembership(value, functionSets):
    value = float(value)
    mv = []
    for fsi in range(len(functionSets)):
        membershipValue = 0
        if fsi == 0:
            if value <= functionSets[fsi][2] and value >= functionSets[fsi][0]:
                membershipValue = 1
            if value <= functionSets[fsi][3] and value >= functionSets[fsi][2]:
                if (functionSets[fsi][3]-functionSets[fsi][2]) == 0:
                    membershipValue = 1
                else:
                    membershipValue = (
                        functionSets[fsi][3]-value)/(functionSets[fsi][3]-functionSets[fsi][2])
        if fsi == len(functionSets)-1:
            if value <= functionSets[fsi][3] and value >= functionSets[fsi][1]:
                membershipValue = 1
            if value <= functionSets[fsi][1] and value >= functionSets[fsi][0]:
                if (functionSets[fsi][1]-functionSets[fsi][0]) == 0:
                    membershipValue = 1
                else:
                    membershipValue = (
                        value - functionSets[fsi][0])/(functionSets[fsi][1]-functionSets[fsi][0])
        if fsi != len(functionSets)-1 and fsi != 0:
            if value <= functionSets[fsi][2] and value >= functionSets[fsi][1]:
                membershipValue = 1
            if value <= functionSets[fsi][3] and value >= functionSets[fsi][2]:
                if (functionSets[fsi][3]-functionSets[fsi][2]) == 0:
                    membershipValue = 1
                else:
                    membershipValue = (
                        functionSets[fsi][3]-value)/(functionSets[fsi][3]-functionSets[fsi][2])
            if value <= functionSets[fsi][1] and value >= functionSets[fsi][0]:
                if (functionSets[fsi][1]-functionSets[fsi][0]) == 0:
                    membershipValue = 1
                else:
                    membershipValue = (
                        value - functionSets[fsi][0])/(functionSets[fsi][1]-functionSets[fsi][0])
        mv.append(membershipValue)
    return mv


def calculateIndirectY(X, newFactors):
    Y = 0
    mu = 0
    for variant in itertools.product(*X):
        print("variant")
        aggrW = 0
        muY = 1
        Yasterisk = 0
        for v in range(len(variant)):
            aggrW = aggrW + \
                float(newFactors[v]["influence"][variant[v]["eval"]])
            muY = min(muY, variant[v]["evalValue"], variant[v]["coef"])

        for v in range(len(variant)):
            Yasterisk = Yasterisk + \
                variant[v]["value"] * \
                newFactors[v]["influence"][variant[v]["eval"]]/aggrW
        Y = Y+Yasterisk*muY
        mu = mu+muY
    if mu != 0:
        Y = Y/mu
    else:
        Y = 0
    return Y


# @app.route('/clusters', methods=['POST'])
# async def clusters():
#     if request.method == 'POST':
#         content = request.json
#         outliers = findOutliersIn(content["data"])
#         l0 = map(lambda x: [x], content["data"][0])


# @app.route('/fuzzy', methods=['POST'])
# def fuzzy():
#     if request.method == 'POST':
#         content = request.json
#         result = []
#         for fs in content["data"]:


def fss(combined, m, n_clusters):
    fss = []
    for i in range(combined.argmax(axis=1).size):
        mean_dists = []
        myIndex = m.argmax(axis=1)[i]
        for ci in range(n_clusters):
            filter_arr = []
            for mi in range(combined.argmax(axis=1).size):
                if m.argmax(axis=1)[mi] == ci:
                    filter_arr.append(True)
                else:
                    filter_arr.append(False)
            newarr = combined[filter_arr]
            mean = 0
            for element in newarr:
                dst = distance.euclidean(combined[i], element)
                mean = mean+dst
            if (myIndex == ci):
                mean = mean/(len(newarr)-1)
            else:
                mean = mean/len(newarr)
            mean_dists.append(mean)
        a = mean_dists[myIndex]
        b = np.min(mean_dists[0:myIndex]+mean_dists[myIndex+1:])
        s = 0
        if a > b:
            s = (b-a)/a
        else:
            s = (b-a)/b
        fss.append(s)
    meanu = 0
    meanl = 0
    for fs in range(combined.argmax(axis=1).size):
        max1 = np.max(m[fs])
        qmax1 = np.argmax(m[fs])
        nar = np.delete(m[fs], qmax1, 0)
        max2 = np.max(nar)
        meanl = meanl+(max1-max2)
        meanu = meanu+(max1-max2)*fss[fs]
    return meanu/meanl


def xiebeni(combined, ma, n_clusters, fcm_centers):
    fss = []
    m = 2
    fu = 0
    fl = 0
    for i in range(combined.argmax(axis=1).size):
        for ci in range(n_clusters):
            mu = ma[i][ci]
            mu = pow(mu, m)
            dst = distance.euclidean(combined[i], fcm_centers[ci])
            dst = pow(dst, 2)
            fu = fu+dst*mu
    min = 9999999999
    for ci in range(n_clusters):
        for cj in range(n_clusters):
            dst = distance.euclidean(fcm_centers[ci], fcm_centers[cj])
            dst = pow(dst, 2)
            if (dst < min and dst != 0):
                min = dst
    fl = min*combined.argmax(axis=1).size
    return fu/fl


def fusu(combined, ma, n_clusters, fcm_centers):
    fss = []
    m = 2
    f1 = 0
    f2 = 0
    for i in range(combined.argmax(axis=1).size):
        for ci in range(n_clusters):
            mu = ma[i][ci]
            mu = pow(mu, m)
            dst = distance.euclidean(combined[i], fcm_centers[ci])
            dst = pow(dst, 2)
            f1 = f1+dst*mu
    mean = np.mean(combined, axis=0)
    for i in range(combined.argmax(axis=1).size):
        for ci in range(n_clusters):
            mu = ma[i][ci]
            mu = pow(mu, m)
            dst = distance.euclidean(fcm_centers[ci], mean)
            dst = pow(dst, 2)
            f2 = f2+dst*mu
    return f1-f2
