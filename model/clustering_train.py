#底层封装好的c++
def train(self, x, weights=None, init_centroids=None):
        """Perform k-means clustering.
        On output of the function call:   函数调用的输出
        - the centroids are in the centroids field of size (`k`, `d`).
        - the objective value at each iteration is in the array obj (size `niter`)
        - detailed optimization statistics are in the array iteration_stats.
        Parameters
        ----------
        x : array_like  array_like是类数组（伪数组）
            Training vectors, shape (n, d), `dtype` must be float32 and n should
            be larger than the number of clusters `k`.
        weights : array_like
            weight associated to each vector, shape `n`
        init_centroids : array_like
            initial set of centroids, shape (n, d)
        Returns
        -------
        final_obj: float
            final optimization objective
        """
        n, d = x.shape
        assert d == self.d

        if self.cp.__class__ == ClusteringParameters:
            # regular clustering
            clus = Clustering(d, self.k, self.cp)
            if init_centroids is not None:
                nc, d2 = init_centroids.shape
                assert d2 == d
                copy_array_to_vector(init_centroids.ravel(), clus.centroids)
            if self.cp.spherical:
                self.index = IndexFlatIP(d)
            else:
                self.index = IndexFlatL2(d)
            if self.gpu:
                self.index = index_cpu_to_all_gpus(self.index, ngpu=self.gpu)
            clus.train(x, self.index, weights)
        else:
            # not supported for progressive dim
            assert weights is None
            assert init_centroids is None
            assert not self.cp.spherical
            clus = ProgressiveDimClustering(d, self.k, self.cp)
            if self.gpu:
                fac = GpuProgressiveDimIndexFactory(ngpu=self.gpu)
            else:
                fac = ProgressiveDimIndexFactory()
            clus.train(n, swig_ptr(x), fac)

        centroids = vector_float_to_array(clus.centroids)

        self.centroids = centroids.reshape(self.k, d)
        stats = clus.iteration_stats
        stats = [stats.at(i) for i in range(stats.size())]
        self.obj = np.array([st.obj for st in stats])
        # copy all the iteration_stats objects to a python array
        stat_fields = 'obj time time_search imbalance_factor nsplit'.split()
        self.iteration_stats = [
            {field: getattr(st, field) for field in stat_fields}
            for st in stats
        ]
        return self.obj[-1] if self.obj.size > 0 else 0.0


def copy_array_to_vector(a, v):
    """ copy a numpy array to a vector """
    n, = a.shape#数组的shape
    classname = v.__class__.__name__
    assert classname.endswith('Vector')
    dtype = np.dtype(vector_name_map[classname[:-6]])
    assert dtype == a.dtype, (
        'cannot copy a %s array to a %s (should be %s)' % (
            a.dtype, classname, dtype))
    v.resize(n)
    if n > 0:
        memcpy(v.data(), swig_ptr(a), a.nbytes)


    }


"""
compute_centroids的作用是计算每个簇的所有向量的总和，以及向量个数，得到每一簇的均值，即为新的聚类中心。
为了提升计算的速度，将簇按照线程数分段，每一个线程计算对应分段的簇。
举个例子，现在有10个线程，100个簇，那么0号线程计算0-9号簇，1号线程计算10-19号簇，以此类推。

"""
#计算中心点
void compute_centroids (size_t d, size_t k, size_t n,
                       size_t k_frozen,
                       const uint8_t * x, const Index *codec,
                       const int64_t * assign,
                       const float * weights,
                       float * hassign,
                       float * centroids)
{
    k -= k_frozen;
    centroids += k_frozen * d;
 
    memset (centroids, 0, sizeof(*centroids) * d * k); // 清零
 
    size_t line_size = codec ? codec->sa_code_size() : d * sizeof (float); //每一个向量的size
 
#pragma omp parallel  //并行计算，将centroids分段，每个线程只计算对应的分段
    {
        int nt = omp_get_num_threads(); //获取总的线程数
        int rank = omp_get_thread_num(); //获取当前线程id
 
        // this thread is taking care of centroids c0:c1
        size_t c0 = (k * rank) / nt; 
        size_t c1 = (k * (rank + 1)) / nt;
        std::vector<float> decode_buffer (d);
 
        for (size_t i = 0; i < n; i++) {
            int64_t ci = assign[i]; //获取离第i个向量最近的聚类中心id（ci）
            if (ci >= c0 && ci < c1)  { //只计算c0-c1的
                float * c = centroids + ci * d;
                const float * xi;
                xi = reinterpret_cast<const float*>(x + i * line_size);
                
                    hassign[ci] += 1.0; //ci簇的向量数加1
                    for (size_t j = 0; j < d; j++) {
                        c[j] += xi[j]; //获取ci簇每个维度上的总和
                    }
            }
        }
 
    }
 
#pragma omp parallel for //并行计算每个簇的均值，得到新的聚类中心
    for (idx_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) {
            continue;
        }
        float norm = 1 / hassign[ci];
        float * c = centroids + ci * d;
        for (size_t j = 0; j < d; j++) {
            c[j] *= norm;
        }
    }
}
"""
split_cluster的作用是找出数量为0的簇，
并找出一个较大的簇，将其平均分成两份，并更新两个小簇对应的中心点。
""" 

int split_clusters (size_t d, size_t k, size_t n,
                    size_t k_frozen,
                    float * hassign,
                    float * centroids)
{
    /* Take care of void clusters */
    size_t nsplit = 0;
    RandomGenerator rng (1234);
    for (size_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) { /* 数量为0的簇，需要找一个大粗分割 */
            size_t cj;
            for (cj = 0; 1; cj = (cj + 1) % k) {
                /* probability to pick this cluster for split */
                float p = (hassign[cj] - 1.0) / (float) (n - k);
                float r = rng.rand_float ();
                if (r < p) {
                    break; /* 找到一个分割的大簇 */
                }
            }
            //将大簇中心copy给小簇
            memcpy (centroids+ci*d, centroids+cj*d, sizeof(*centroids) * d);
 
            /* 通过对两个相同的中心添加反向扰动，从而分成两个中心 */
            for (size_t j = 0; j < d; j++) {
                if (j % 2 == 0) {
                    centroids[ci * d + j] *= 1 + EPS;
                    centroids[cj * d + j] *= 1 - EPS;
                } else {
                    centroids[ci * d + j] *= 1 - EPS;
                    centroids[cj * d + j] *= 1 + EPS;
                }
            }
 
            /* 更新对应簇的数量 */
            hassign[ci] = hassign[cj] / 2;
            hassign[cj] -= hassign[ci];
            nsplit++;
        }
    }
    return nsplit;
}
"""
通过compute&split这两个函数，可以得到更新后的中心，再将flat index 清零，并将新的中心点添加到index，作为下一次迭代的搜索索引。

这样直到迭代次数（60）结束，选出距离偏差err最小的一次训练结果作为聚类的最后结果。
"""
