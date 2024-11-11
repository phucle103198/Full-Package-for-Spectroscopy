from WaveSelect.Lar import Lar
from WaveSelect.Spa import SPA
from WaveSelect.Uve import UVE
from WaveSelect.Cars import CARS_Cloud
from WaveSelect.Pca import Pca
from WaveSelect.GA import GA
from sklearn.model_selection import train_test_split

def SpctrumFeatureSelcet(method, X, y):
    if method == "None":
        X_Feature = X
    elif method== "Cars":
        Featuresecletidx = CARS_Cloud(X, y)
        X_Feature = X[:, Featuresecletidx]
    elif method == "Lars":
        Featuresecletidx = Lar(X, y)
        X_Feature = X[:, Featuresecletidx]
    elif method == "Uve":
        Uve = UVE(X, y, 7)
        Uve.calcCriteria()
        Uve.evalCriteria(cv=5)
        Featuresecletidx = Uve.cutFeature(X)
        X_Feature = Featuresecletidx[0]
    elif method == "Spa":
        Xcal, Xval, ycal, yval = train_test_split(X, y, test_size=0.2)
        Featuresecletidx = SPA().spa(
            Xcal= Xcal, ycal=ycal, m_min=8, m_max=50, Xval=Xval, yval=yval, autoscaling=1)
        X_Feature = X[:, Featuresecletidx]
    elif method == "GA":
        Featuresecletidx = GA(X, y, 10)
        X_Feature = X[:, Featuresecletidx]
    elif method == "Pca":
        X_Feature = Pca(X)
    else:
        print("no this method of SpctrumFeatureSelcet!")

    return X_Feature, y
