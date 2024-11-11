from sklearn import linear_model
import numpy as np

def Lar(X, y, nums=40):
    Lars = linear_model.Lars()
    Lars.fit(X, y)
    corflist = np.abs(Lars.coef_)

    corf = np.asarray(corflist)
    SpectrumList = corf.argsort()[-1:-(nums+1):-1]
    SpectrumList = np.sort(SpectrumList)

    return SpectrumList
