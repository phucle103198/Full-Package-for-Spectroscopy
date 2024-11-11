from sklearn.cluster import KMeans
import numpy as np

def Kmeans(data, n_clusters=10, iter_num=30):

    cluster = KMeans(n_clusters=n_clusters, random_state=0, max_iter=iter_num)
    cluster.fit(data)
    label = cluster.labels_  

    return label

class FCM:
    def __init__(self, data, clust_num, iter_num=10, m=2) :
        self.data = data
        self.cnum = clust_num
        self.sample_num=data.shape[0]
        self.m = m
        self.dim = data.shape[-1] 
        Jlist=[]  

        U = self.Initial_U(self.sample_num, self.cnum)
        for i in range(0, iter_num): 
            C = self.Cen_Iter(self.data, U, self.cnum)
            U = self.U_Iter(U, C)
            J = self.J_calcu(self.data, U, C)  
            Jlist = np.append(Jlist, J)
        self.label = np.argmax(U, axis=0) 
        self.Clast = C    
        self.Jlist = Jlist 

    def Initial_U(self, sample_num, cluster_n):
        U = np.random.rand(sample_num, cluster_n)  
        row_sum = np.sum(U, axis=1)  
        row_sum = 1 / row_sum  
        U = np.multiply(U.T, row_sum)  
        return U   # cluster_n*sample_num

    def Cen_Iter(self, data, U, cluster_n, m):
        c_new = np.empty(shape=[0, self.dim])  
        for i in range(0, cluster_n):        
            u_ij_m = U[i, :] ** m  # (sample_num,)
            sum_u = np.sum(u_ij_m)
            ux = np.dot(u_ij_m, data)  # (dim,)
            ux = np.reshape(ux, (1, self.dim))  # (1,dim)
            c_new = np.append(c_new, ux / sum_u, axis=0)  
        return c_new  # cluster_num*dim

    def U_Iter(self, U, c, m):
        for i in range(0, self.cnum):
            for j in range(0, self.sample_num):
                sum = 0
                for k in range(0, self.cnum):
                    temp = (np.linalg.norm(self.data[j, :] - c[i, :]) /
                            np.linalg.norm(self.data[j, :] - c[k, :])) ** (
                                2 / (m - 1))
                    sum = temp + sum
                U[i, j] = 1 / sum

        return U


    def J_calcu(self, data, U, c, m):
        temp1 = np.zeros(U.shape)
        for i in range(0, U.shape[0]):
            for j in range(0, U.shape[1]):
                temp1[i, j] = (np.linalg.norm(data[j, :] - c[i, :])) ** 2 * U[i, j] ** m

        J = np.sum(np.sum(temp1))

        return J

def Fcm(data, n_clusters=10, iter_num=30):

    Fcm = FCM(data, n_clusters, iter_num)
    label =Fcm.U_Iter()

    return  label

def Cluster(method, data):
    if method == "Kmeans":
        label = Kmeans(data)
    if method == "Fcm":
        label = Fcm(data)
    return label



