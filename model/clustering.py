# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#faiss是为稠密向量提供高效相似度搜索和聚类的框架,c++实现，提供python封装调用
import time

import faiss
import numpy as np
from PIL import Image
from PIL import ImageFile
from scipy.sparse import csr_matrix, find
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['PIC', 'Kmeans', 'cluster_assign', 'arrange_clustering']

def preprocess_features(npdata, pca=256):#特征预处理
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output输出的维度
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')#因为faiss框架要求输入都是n*d的矩阵格式，然后float展平到一维npdata数组中

    # Apply PCA-whitening with Faiss     使用PCA来降维
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def make_graph(xb, nnn):
    """Builds a graph of nearest neighbors.
    Args:
        xb (np.array): data
        nnn (int): number of nearest neighbors
    Returns:
        list: for each data the list of ids to its nnn nearest neighbors
        list: for each data the list of distances to its nnn NN
    """
    N, dim = xb.shape

    # we need only a StandardGpuResources per GPU每个GPU只需要一个StandardGpuResources
    res = faiss.StandardGpuResources()

    # L2
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatL2(res, dim, flat_config)#IndexFlatL2是把query和所有base进行比对，选取最近的
    index.add(xb)#把xb数据(矢量)加入到index中，用于下面的索引
    D, I = index.search(xb, nnn + 1)#把xb作为查询向量集合，查找其最近的nnn+1个点
    #search检索时根据使用的 metric(度量标准)类型不同，这里是 L2，前k个最可能属于的类别
    return I, D


def cluster_assign(images_lists):
    """Creates a dataset from clustering, with clusters as labels.从集群创建一个数据集，集群作为标签。
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster对于每个群集，属于该群集的映像索引列表
        dataset (list): initial dataset初始数据集
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels以簇作为标签的数据集
    """
    assert images_lists is not None
    #image_label想象的标签
    pseudolabels = []
    #image_index想象的索引
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
    #image_label_index
    #images_label_index = []
    images_dict = {}
    for j, idx in enumerate(image_indexes):
        pseudolabel = label_to_idx[pseudolabels[j]]
        #images_label_index.append(pseudolabel)
        images_dict[idx] = pseudolabel

    #return image_indexes, images_label_index
    #images_dict = sorted(images_dict.items(), key=lambda obj: obj[0])
    return images_dict


def run_kmeans(x, nmb_clusters, niter, verbose=False):
    """Runs kmeans on 1 GPU.在1个GPU上面运行kmeans聚类
    Args:
        x: data 数据
        nmb_clusters (int): number of clusters 聚类的数量：23
    Returns:
        list: ids of data in each cluster 每个聚类中数据的id
    """
    
    n_data, d = x.shape#输入数据的形状
    #print(d) 128 维度
    #print(n_data) 225033
    # faiss implementation of k-means 用faiss实现K-means
    clus = faiss.Clustering(d, nmb_clusters)#d是维度，nmb_clusters是聚类数=23

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)#随机生成

    clus.niter = niter#clus_niter=60训练的迭代次数
    clus.max_points_per_centroid = 10000000#限制数据集大小
    res = faiss.StandardGpuResources()#使用单个gpu
    #flat_config 是个ID. if you have 3 GPUs, flat_configs maybe 0,1,2
    flat_config = faiss.GpuIndexFlatConfig()#构建一个空量化器
    flat_config.useFloat16 = False#useFloat16是一个布尔型索引
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)#暴力检索，IndexFlatL2(d)建立索引,d=维度,flat_config设置只在一个GPU上运行
    """
    实现 K-means 聚类的类，提供train ，需要训练数据和 index（用于 search 最近的向量），
    结果得到训练数据的类中心向量，(如果是量化的向量，那么还需要提供量化使用的 index codec)
    这里去除量化的部分，只看 float 数据

    核心代码包括如下部分：

    1.search过程，将聚类中心作为底库加入到 index 中，并对训练数据做 search，得到assign
    2.计算新的聚类中心，计算新的聚类中心的代码在     中，
    具体就是对于相同的类别的向量，将向量的均值作为新的中心，在实现上，利用 openmp 进行了并行优化
    重复以上两步，就可以得到最优的聚类中心

    """
    # 开始训练
    clus.train(x, index)#训练聚类 x就是数据,train的定义在上面
    _, I = index.search(x, 1)#不知道什么卵用
    losses = faiss.vector_to_array(clus.obj)#强制转换数据类型向量为数组
    if verbose:#在train_cluster_bert_mrc.py中设置的是true
        print('k-means聚类的演变过程: {0}'.format(losses))#训练的迭代次数是60,得到训练数据的类中心向量
        """
        k-means loss evolution: [209562.39  122201.42  112577.74  108027.72  106978.5   105880.63
 105213.98  104836.48  104527.93  104361.73  104228.266 104075.58
 103888.984 103760.98  103694.35  103654.28  103623.14  103597.734
 103578.86  103562.37  103546.54  103529.55  103514.06  103499.1
 103486.16  103470.68  103456.97  103446.164 103432.85  103421.836
 103411.03  103400.32  103388.72  103376.    103361.41  103345.61
 103329.79  103311.84  103292.95  103272.53  103251.875 103228.72
 103208.98  103189.125 103170.61  103153.91  103134.78  103117.66
 103099.875 103083.016 103062.56  103042.73  103022.71  103001.73
 102983.766 102964.625 102949.695 102937.39  102926.5   102914.99 ]
K-means evolution结束了.....................
        """
        print("K-means evolution结果如上！训练结束！")
#演变到质心不再发生改变
    return [int(n[0]) for n in I], losses[-1]#a=[1,2,3,4,5]a[-1]=5表最后一位

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans(object):
    def __init__(self, kk):
        self.k = kk#聚类中心 质心
        
        #kk=23
    def plot_embedding_mergelabel(self, data, label, view_number, epoch):
        label = np.array(label)
        """
        #sample_cluster = np.random.choice(21,6)
        sample_index = np.random.choice(99, 100)
        indexs=[]
        for i in range(6):
            l=len(index_list[i])
            if l<100:
                indexs.extend(index_list[i][:min([50, l])])
                continue
            indexs.extend([index_list[i][j] for j in sample_index])
        
        """

        x_min, x_max = np.min(data, 0), np.max(data, 0)#找到每一列中的最小/大值
        data = (data - x_min) / (x_max - x_min)
        #print("var_of_totaldata:",np.var(data))#0.017024202,无输出
        
        #row_sums = np.linalg.norm(data, axis=1)
        #data = data / row_sums[:, np.newaxis]
        index_dict = {}
        cluster_dict = {}
        for i in range(self.k):#循环每一个质心（23）
            v_x=self.images_lists[i][:100]#切片操作，获取一个序列
            v=np.var(data[v_x,:])#data[v_x,:]表示在v_x个数组（维）中取全部数据
            #var求方差：各个数据与平均数之差的平方的平均数。
            index_dict[v]=v_x
            #print("输出index_dict[v].........................")
            #print(index_dict[v])#1~204596: 10
            cluster_dict[v]=i



        dict2 = sorted(cluster_dict, reverse=False)#升序排列
        indexs = []

        """
        for i in range(self.k):
            v=dict2[i]
            v_x=[]
            for j in range(i, self.k):
                v_x.extend(index_dict[dict2[j]])
            compare_v=np.var(data[v_x,:])
            print("compare_v" + str(compare_v) + "<" + str(v))
            if (compare_v-v)<view_number:
                view_number=i
                print("view_number:" + str(i))
                break
        view_number+=1
        """
        #image_list_temp = [[] for i in range(view_number + 1)]
        for i in range(self.k):#只要i在23个簇中，循环整数列表
            #("输出self.k............................")
            #print(self.k) 5,12330 5,,12356  5,122384 5,122388 5,122389 5,122404 122405
            v=dict2[i]#将i升序排列
            label[index_dict[v]]=i
            #print(str(v)+"\t"+str(len(index_dict[v])))#后一个[23] 100
            
            if i < view_number:
                indexs.extend(index_dict[v])
                ##image_list_temp[i]=self.images_lists[cluster_dict[v]]
            #else:
                #indexs.extend(index_dict[v][:10])
                #label[index_dict[v]] = view_number
                #image_list_temp[view_number].extend(self.images_lists[cluster_dict[v]])

        tsne = TSNE(n_components=3, init='pca', random_state=25)
        data=data[indexs, :]
        label = label[indexs]
        data = tsne.fit_transform(data)

        
        color=np.vstack((plt.cm.Set1([i for i in range(9)]),plt.cm.Set2([i for i in range(8)]),plt.cm.Set3([i for i in range(12)])))
        #color = plt.cm.Set3([i for i in range(view_number+1)])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=np.array([list(color[label[i]]) for i in range(len(label))]))
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')
        ax.grid(False)
        plt.axis('off')
        plt.savefig("data/saved_fig/cluster-" + str(self.k) + "-" + str(view_number) + "-" + "-" + str(epoch))
        #ax.view_init(45, 120)
        #plt.title(title)
        #plt.show()
        #return fig


    def plot_embedding(self, data, label):
        label = np.array(label)

        #x_min, x_max = np.min(data, 0), np.max(data, 0)
        #data = (data - x_min) / (x_max - x_min)
        #print("var_of_totaldata:", np.var(data))#不画图没输出
        
        index_dict = {}
        cluster_dict = {}
        for i in range(self.k):#只要i在23个簇中
            v_x = self.images_lists[i][:]
            v = np.var(data[v_x, :])
            index_dict[v] = v_x
            cluster_dict[v] = i

        sorted_var = sorted(cluster_dict, reverse=False)
        indexs = []


        image_list_temp = [[] for i in range(self.k)]
        for i in range(self.k):
            v = sorted_var[i]
            label[index_dict[v]] = i
            
            #print(str(v) + "\t" + str(len(index_dict[v])))#不输出
            
            image_list_temp[i] = self.images_lists[cluster_dict[v]]

        self.images_lists = image_list_temp
        image_list_temp = []
        sorted_var = 1/np.array(sorted_var)
       
       #print(softmax(sorted_var))#啥也没输出
        
        return sorted_var


    def draw_cluster(self):
        np.random.seed(25)
        pubmed_class_label = ['biological_process_involves_gene_product', 'inheritance_type_of',
                              'is_normal_tissue_origin_of_disease', 'ingredient_of',
                              'is_primary_anatomic_site_of_disease', 'gene_found_in_organism', 'occurs_in',
                              'causative_agent_of', 'classified_as', 'gene_plays_role_in_process']

        file_name = "most_val_pubmed_features"

        val_fea = np.load(file_name + ".npy")

        sample_index = np.random.choice(99, 20)
        val_fea = val_fea[:, sample_index, :]

        val_fea = np.reshape(val_fea, [-1, 230])

        tsne_adv = TSNE(n_components=2, init="pca", random_state=25)
        val_fea_tsne = tsne_adv.fit_transform(val_fea)

        val_fea_tsne = val_fea_tsne.reshape(10, 20, 2)

        plt.xticks([])
        plt.yticks([])

        legend_record = []
        for i in range(10):
            leg = plt.scatter(val_fea_tsne[i, :, 0], val_fea_tsne[i, :, 1])
            legend_record.append(leg)

        plt.legend(handles=legend_record, labels=pubmed_class_label)
        #plt.show()
        plt.savefig(file_name + ".png", pad_inches=0.1, bbox_inches='tight')

    def cluster(self, data, view_number, cluster_layer, pca_dim, niter, epoch, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data, pca_dim)

        # cluster the data
        I, loss = run_kmeans(xb, self.k, niter, verbose)

        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):#len(data)是225033
            self.images_lists[I[i]].append(i)

        #draw
        """画出的图是这部分定义的
        print("&&&&&Begin Drawing&&&&&")
        #var=self.plot_embedding(xb, I)
        self.plot_embedding_mergelabel(xb, I, 4, epoch)
        self.plot_embedding_mergelabel(xb, I, 5, epoch)
        self.plot_embedding_mergelabel(xb, I, 6, epoch)
        self.plot_embedding_mergelabel(xb, I, 7, epoch)
        self.plot_embedding_mergelabel(xb, I, 8, epoch)
        print("&&&&&End Drawing&&&&&")
        """

        var=[]
        #x_min, x_max=np.min(data,0), np.max(data,0)
        #data=(data-x_min) / (x_max-x_min)
        for i in range(self.k):#i进行循环到23
            var.append(np.var(data[self.images_lists[i], :]))#np.var求方差，方差越大，越离散
        print("经过训练得到簇类方差为：")
        print(var)
        """
        
[1.0850456, 1.3453761, 1.4214762, 92.86358, 1.9389765, 2.100384, 0.8412168, 129.2305, 0.6077103, 1.965053, 83357.11, 3.8375452, 2.2738364, 7.793381, 12480.096, 0.9345613, 4.0348, 5.0104966, 4.012915, 5.6007977, 4.457743, 6.047816, 3.0197437]
        """
        
        var=1/np.array(var)
        print("将方差进行softmax归一化得到：")
        print(softmax(var))#softmax归一化函数，1）预测的概率为非负数；2）各种预测结果概率之和等于1。
        """
        最后输出的是softmax(var)
[0.06318564 0.05286509 0.05080276 0.0254121  0.04210611 0.04046997
 0.08253405 0.0253352  0.13031802 0.04181892 0.02514021 0.03262373
 0.03902655 0.02858182 0.02514193 0.07329392 0.03221076 0.0306931
 0.03225433 0.03005421 0.03146216 0.02966019 0.03500919]
        """


        print("输出聚类标签为以下23类：")        #输入287314，
        for i in range(len(self.images_lists)):#一共211933，
            
            print("Number of Label-"+str(i)+": ", len(self.images_lists[i]))#str(i)=0~22

        if verbose:#输出
            print("输出k-means的时间：")
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss, var
"""
开始输出聚类标签.......................................
Number of Label-0:  13100
开始输出聚类标签.......................................
Number of Label-1:  10556
开始输出聚类标签.......................................
Number of Label-2:  10659
开始输出聚类标签.......................................
Number of Label-3:  5505
开始输出聚类标签.......................................
Number of Label-4:  8116
开始输出聚类标签.......................................
Number of Label-5:  12754
开始输出聚类标签.......................................
Number of Label-6:  11771
开始输出聚类标签.......................................
Number of Label-7:  5500
开始输出聚类标签.......................................
Number of Label-8:  7148
开始输出聚类标签.......................................
Number of Label-9:  10903
开始输出聚类标签.......................................
Number of Label-10:  18257
开始输出聚类标签.......................................
Number of Label-11:  8994
开始输出聚类标签.......................................
Number of Label-12:  3866
开始输出聚类标签.......................................
Number of Label-13:  6179
开始输出聚类标签.......................................
Number of Label-14:  2778
开始输出聚类标签.......................................
Number of Label-15:  14971
开始输出聚类标签.......................................
Number of Label-16:  10527
开始输出聚类标签.......................................
Number of Label-17:  5570
开始输出聚类标签.......................................
Number of Label-18:  13138
开始输出聚类标签.......................................
Number of Label-19:  7695
开始输出聚类标签.......................................
Number of Label-20:  11989
开始输出聚类标签.......................................
Number of Label-21:  11495
开始输出聚类标签.......................................
Number of Label-22:  13562
开始输出k-means的时间..................................
k-means time: 17 s
"""

def make_adjacencyW(I, D, sigma):
    """Create adjacency matrix with a Gaussian kernel.用高斯核生成连接矩阵
    Args:
        I (numpy array): for each vertex the ids to its nnn linked vertices
                  + first column of identity.对于每个顶点，将ID添加到其nnn链接的顶点+标识的第一列。
        D (numpy array): for each data the l2 distances to its nnn linked vertices
                  + first column of zeros.对于每个数据，l2到其nnn链接顶点的距离+第一列零。
        sigma (float): Bandwidth of the Gaussian kernel.

    Returns:
        csr_matrix: affinity matrix of the graph.
    """
    V, k = I.shape
    k = k - 1
    indices = np.reshape(np.delete(I, 0, 1), (1, -1))
    indptr = np.multiply(k, np.arange(V + 1))

    def exp_ker(d):
        return np.exp(-d / sigma**2)

    exp_ker = np.vectorize(exp_ker)
    res_D = exp_ker(D)
    data = np.reshape(np.delete(res_D, 0, 1), (1, -1))
    adj_matrix = csr_matrix((data[0], indices[0], indptr), shape=(V, V))
    return adj_matrix


def run_pic(I, D, sigma, alpha):
    """Run PIC algorithm"""
    a = make_adjacencyW(I, D, sigma)
    graph = a + a.transpose()
    cgraph = graph
    nim = graph.shape[0]

    W = graph
    t0 = time.time()

    v0 = np.ones(nim) / nim

    # power iterations
    v = v0.astype('float32')

    t0 = time.time()
    dt = 0
    for i in range(200):
        vnext = np.zeros(nim, dtype='float32')

        vnext = vnext + W.transpose().dot(v)

        vnext = alpha * vnext + (1 - alpha) / nim
        # L1 normalize
        vnext /= vnext.sum()
        v = vnext

        if i == 200 - 1:
            clust = find_maxima_cluster(W, v)

    return [int(i) for i in clust]


def find_maxima_cluster(W, v):
    n, m = W.shape
    assert (n == m)
    assign = np.zeros(n)
    # for each node
    pointers = list(range(n))
    for i in range(n):
        best_vi = 0
        l0 = W.indptr[i]
        l1 = W.indptr[i + 1]
        for l in range(l0, l1):
            j = W.indices[l]
            vi = W.data[l] * (v[j] - v[i])
            if vi > best_vi:
                best_vi = vi
                pointers[i] = j
    n_clus = 0
    cluster_ids = -1 * np.ones(n)
    for i in range(n):
        if pointers[i] == i:
            cluster_ids[i] = n_clus
            n_clus = n_clus + 1
    for i in range(n):
        # go from pointers to pointers starting from i until reached a local optim
        current_node = i
        while pointers[current_node] != current_node:
            current_node = pointers[current_node]

        assign[i] = cluster_ids[current_node]
        assert (assign[i] >= 0)
    return assign


class PIC(object):
    """Class to perform Power Iteration Clustering on a graph of nearest neighbors.
        Args:
            args: for consistency with k-means init
            sigma (float): bandwidth of the Gaussian kernel (default 0.2)
            nnn (int): number of nearest neighbors (default 5)
            alpha (float): parameter in PIC (default 0.001)
            distribute_singletons (bool): If True, reassign each singleton to
                                      the cluster of its closest non
                                      singleton nearest neighbors (up to nnn
                                      nearest neighbors).
        Attributes:
            images_lists (list of list): for each cluster, the list of image indexes
                                         belonging to this cluster  对于每个群集，属于该群集的映像索引列表
    """

    def __init__(self, args=None, sigma=0.2, nnn=5, alpha=0.001, distribute_singletons=True):
        self.sigma = sigma
        self.alpha = alpha
        self.nnn = nnn
        self.distribute_singletons = distribute_singletons

    def cluster(self, data, verbose=False):
        end = time.time()

        # preprocess the data
        xb = preprocess_features(data)

        # construct nnn graph
        I, D = make_graph(xb, self.nnn)

        # run PIC
        clust = run_pic(I, D, self.sigma, self.alpha)
        images_lists = {}
        for h in set(clust):
            images_lists[h] = []
        for data, c in enumerate(clust):
            images_lists[c].append(data)

        # allocate singletons to clusters of their closest NN not singleton
        if self.distribute_singletons:
            clust_NN = {}
            for i in images_lists:
                # if singleton
                if len(images_lists[i]) == 1:
                    s = images_lists[i][0]
                    # for NN
                    for n in I[s, 1:]:
                        # if NN is not a singleton
                        if not len(images_lists[clust[n]]) == 1:
                            clust_NN[s] = n
                            break
            for s in clust_NN:
                del images_lists[clust[s]]
                clust[s] = clust[clust_NN[s]]
                images_lists[clust[s]].append(s)

        self.images_lists = []
        for c in images_lists:
            self.images_lists.append(images_lists[c])

        if verbose:#因为设置的是false，所以没输出
            print("最后输出pic time..............................")
            print('pic time: {0:.0f} s'.format(time.time() - end))
        return 0

