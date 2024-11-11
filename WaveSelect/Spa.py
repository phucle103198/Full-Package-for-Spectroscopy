import pandas as pd
import numpy as np
from scipy.linalg import qr, inv, pinv
import scipy.stats
import scipy.io as scio
# from progress.bar import Bar
from matplotlib import pyplot as plt


class SPA:

    def _projections_qr(self, X, k, M):

        X_projected = X.copy()
        norms = np.sum((X ** 2), axis=0)
        norm_max = np.amax(norms)
        X_projected[:, k] = X_projected[:, k] * 2 * norm_max / norms[k]

        _, __, order = qr(X_projected, 0, pivoting=True)

        return order[:M].T

    def _validation(self, Xcal, ycal, var_sel, Xval=None, yval=None):
        N = Xcal.shape[0]  
        if Xval is None:  
        else:
            NV = Xval.shape[0]  

        yhat = e = None

        if NV > 0:
            Xcal_ones = np.hstack(
                [np.ones((N, 1)), Xcal[:, var_sel].reshape(N, -1)])

            b = np.linalg.lstsq(Xcal_ones, ycal, rcond=None)[0]

            np_ones = np.ones((NV, 1))
            Xval_ = Xval[:, var_sel]
            X = np.hstack([np.ones((NV, 1)), Xval[:, var_sel]])
            yhat = X.dot(b)

            e = yval - yhat
        else:
            yhat = np.zeros((N, 1))
            for i in range(N):
                cal = np.hstack([np.arange(i), np.arange(i + 1, N)])
                X = Xcal[cal, :][:, var_sel.astype(np.int)]
                y = ycal[cal]
                xtest = Xcal[i, var_sel]
                # ytest = ycal[i]
                X_ones = np.hstack([np.ones((N - 1, 1)), X.reshape(N - 1, -1)])
                b = np.linalg.lstsq(X_ones, y, rcond=None)[0]
                yhat[i] = np.hstack([np.ones(1), xtest]).dot(b)
            e = ycal - yhat

        return yhat, e

    def spa(self, Xcal, ycal, m_min=1, m_max=None, Xval=None, yval=None, autoscaling=1):


        assert (autoscaling == 0 or autoscaling == 1),

        N, K = Xcal.shape

        if m_max is None:
            if Xval is None:
                m_max = min(N - 1, K)
            else:
                m_max = min(N - 2, K)

            assert (m_max < min(N - 1, K)), "m_max"

        normalization_factor = None
        if autoscaling == 1:
            normalization_factor = np.std(
                Xcal, ddof=1, axis=0).reshape(1, -1)[0]
        else:
            normalization_factor = np.ones((1, K))[0]

        Xcaln = np.empty((N, K))
        for k in range(K):
            x = Xcal[:, k]
            Xcaln[:, k] = (x - np.mean(x)) / normalization_factor[k]

        SEL = np.zeros((m_max, K))

        # with Bar('Projections :', max=K) as bar:
        for k in range(K):
            SEL[:, k] = self._projections_qr(Xcaln, k, m_max)

        PRESS = float('inf') * np.ones((m_max + 1, K))

        # with Bar('Evaluation of variable subsets :', max=(K) * (m_max - m_min + 1)) as bar:
        for k in range(K):
            for m in range(m_min, m_max + 1):
                var_sel = SEL[:m, k].astype(np.int)
                _, e = self._validation(Xcal, ycal, var_sel, Xval, yval)
                PRESS[m, k] = np.conj(e).T.dot(e)

        #            bar.next()

        PRESSmin = np.min(PRESS, axis=0)
        m_sel = np.argmin(PRESS, axis=0)
        k_sel = np.argmin(PRESSmin)

        var_sel_phase2 = SEL[:m_sel[k_sel], k_sel].astype(np.int)

        Xcal2 = np.hstack([np.ones((N, 1)), Xcal[:, var_sel_phase2]])
        b = np.linalg.lstsq(Xcal2, ycal, rcond=None)[0]
        std_deviation = np.std(Xcal2, ddof=1, axis=0)

        relev = np.abs(b * std_deviation.T)
        relev = relev[1:]

        index_increasing_relev = np.argsort(relev, axis=0)
        index_decreasing_relev = index_increasing_relev[::-1].reshape(1, -1)[0]

        PRESS_scree = np.empty(len(var_sel_phase2))
        yhat = e = None
        for i in range(len(var_sel_phase2)):
            var_sel = var_sel_phase2[index_decreasing_relev[:i + 1]]
            _, e = self._validation(Xcal, ycal, var_sel, Xval, yval)

            PRESS_scree[i] = np.conj(e).T.dot(e)

        RMSEP_scree = np.sqrt(PRESS_scree / len(e))

        PRESS_scree_min = np.min(PRESS_scree)
        alpha = 0.25
        dof = len(e)
        fcrit = scipy.stats.f.ppf(1 - alpha, dof, dof)
        PRESS_crit = PRESS_scree_min * fcrit

        i_crit = np.min(np.nonzero(PRESS_scree < PRESS_crit))
        i_crit = max(m_min, i_crit)

        var_sel = var_sel_phase2[index_decreasing_relev[:i_crit]]

        return var_sel

    def __repr__(self):
        return "SPA()"
