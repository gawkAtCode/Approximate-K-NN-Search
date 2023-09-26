from utils import *
from pynndescent_opt import NNDescent
faiss.omp_set_num_threads(16)

t0=time.perf_counter()

k=100
dim=100

if len(sys.argv)>1:
    source_path=str(sys.argv[1])
else:
    source_path="dummy-data.bin"

X=readBin(source_path)

if X.shape == (10000,100):
    D,I=IVF(X, X, dim = 100, k = 101)
    I=I.astype("int32")
    n_iters,n_jobs,max_candidates,n_neighbors=1,32,101,101
    t1=time.perf_counter()
    index = NNDescent(X, metric="sqeuclidean", n_neighbors=n_neighbors, init_graph=I, init_dist=D, random_state=2023, max_candidates=max_candidates, n_iters=n_iters, n_jobs=n_jobs, verbose=False)
    I2 = index._neighbor_graph[0][:,:100]
    t2=time.perf_counter()
    I2=I2.astype("uint32")
    I2.tofile("output.bin")
else:
    D,I = composite_indexes(X, "IVF1100,PQ100x4fsr,RFlat",nprobe=77,k=340)
    I=I.astype("int32")
    n_iters,n_jobs,max_candidates,n_neighbors=1,32,151,101
    t1=time.perf_counter()
    index = NNDescent(X, metric="sqeuclidean", n_neighbors=n_neighbors, init_graph=I, init_dist=D, random_state=2023, max_candidates=max_candidates, n_iters=n_iters, n_jobs=n_jobs, verbose=False)
    I2 = index._neighbor_graph[0][:,:100]
    t2=time.perf_counter()
    I2=I2.astype("uint32")
    I2.tofile("output.bin")	
    
t3=time.perf_counter()
print("NNDescent(n_iters=%d,n_neighbors=%d,max_candidates=%d,n_jobs=%d):%dm%ds"%(n_iters,n_neighbors,max_candidates,n_jobs,(t2-t1)/60,t2-t1-60*((t2-t1)//60)))		
print("Approximate K-nearest-neighbor Graph Construction (faiss+NNdescent) :%dm%ds"%((t3-t0)/60,t3-t0-60*((t3-t0)//60)))


