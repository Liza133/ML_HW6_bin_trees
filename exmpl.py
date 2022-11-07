import numpy as np
import pandas as pd
import plotly.graph_objects as go


def accuracy(tp,tn,fp,fn):
    return (tp+tn)/(tp+tn+fp+fn)

def presicion(tp,fp):
    return (tp) / (tp + fp)

def recall(tp,fn):
    return (tp) / (tp + fn)

def f1_score(presicion,recall):
    return 2 * (recall*presicion) / (recall + presicion)

def error_1(tn,fp):
    return (fp) / (fp + tn)

def error_2(tp,fn):
    return (fn) / (fn + tp)


N = 500
football_player = np.random.randn(N)*20 + 160# N
basketball_player = np.random.randn(N)*10 + 190# P
threshold_max = 230

predictions = np.concatenate((football_player/threshold_max,basketball_player/threshold_max))
gt = np.concatenate((np.zeros(N),np.ones(N))).astype(int)

sorted_arguments = np.argsort(predictions)[::-1]
predictions = predictions[sorted_arguments]
gt = gt[sorted_arguments]

df = pd.DataFrame({'prediction':predictions,
                           'gt': gt})
tp_ = np.cumsum(gt)
tp_fp = np.arange(len(tp_)) + 1
tp_fn = N
presicion_arr = tp_/tp_fp
recall_arr = tp_/tp_fn

fig = go.Figure(go.Scatter(x=recall_arr,y=presicion_arr))
fig.show()

mrec = np.concatenate(([0.], recall_arr, [1.]))
mpre = np.concatenate(([0.], presicion_arr, [0.]))

for i in range(len(mpre) - 1, 0, -1):
    mpre[i - 1] = max(mpre[i - 1], mpre[i])

fig = go.Figure(go.Scatter(x=mrec,y=mpre))
fig.show()

dots = np.where(mrec[1:] != mrec[:-1])[0]
pr_auc = np.sum((mrec[dots + 1] - mrec[dots]) * mpre[dots + 1])


