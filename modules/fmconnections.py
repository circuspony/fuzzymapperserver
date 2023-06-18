
# @app.route('/regression', methods=['POST'])
# def regression():
#     if request.method == 'POST':
#         content = request.json
#         newRegressions = []

#         for fsi in content["data"]["fsi"]:
#             for fei in content["data"]["fei"]:
#                 X = np.array(list(fsi["values"])).astype(float)
#                 y = np.array(list(fei["values"])).astype(float)
#                 clusterLabels = []


# @app.route('/clustercomp', methods=['POST'])
# def clustercomp():
#     if request.method == 'POST':
#         content = request.json
#         labels1 = content["data"]["iv"]
#         labels2 = content["data"]["ov"]
#         ic = len(labels1[0])
#         oc = len(labels2[0])
#         matrix = [[0]*len(labels2[0]) for i in range(len(labels1[0]))]
#         l = map(lambda x: x.index(max(x)), labels1)
