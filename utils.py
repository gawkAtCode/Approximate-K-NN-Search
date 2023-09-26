import time
import struct
import faiss
import mkl
import numpy as np
mkl.get_max_threads()

def readBin(path):
    data=[]
    with open(path,"rb") as f:
        size=f.read(4)
        size=struct.unpack("i",size)[0]
    X = np.fromfile(path, dtype=np.float32, offset=4)
    X.shape=size,100
    return X
    
def BF(X, query, dim = 100, k = 100):
    index = faiss.IndexFlatL2(dim)
    index.add(X)
    D, I = index.search(query, k)
    return D,I

def IVF(X, query, dim = 100, k = 100):
    nlist = int(np.sqrt(X.shape[0]))
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    index.train(X)
    index.add(X)
    index.nprobe = 5
    D, I = index.search(query, k)
    return D,I

def HNSW(X, dim = 100, k = 100, M = 128, efConstruction = 64, efSearch = 250):
    print("----------------------------------------------------------------")
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = efSearch
    t0=time.perf_counter()
    index.add(X)
    t1=time.perf_counter()
    print("HNSW(M=%d,efConstruction=%d,efSearch=%d) Build:%dm%ds"%(M,efConstruction,efSearch,(t1-t0)/60,t1-t0-60*((t1-t0)//60)))
    t2=time.perf_counter()
    D, I = index.search(X, k)
    t3=time.perf_counter()
    print("HNSW(M=%d,efConstruction=%d,efSearch=%d) Search:%dm%ds"%(M,efConstruction,efSearch,(t3-t2)/60,t3-t2-60*((t3-t2)//60)))
    return D,I

def NSG(X, query, dim = 100, k = 100, R=50):
    print("----------------------------------------------------------------")
    index=faiss.IndexNSGFlat(dim,R)
    t0=time.perf_counter()
    index.add(X)
    t1=time.perf_counter()
    print("NSG(R=%d) Build:%dm%ds"%(R,(t1-t0)/60,t1-t0-60*((t1-t0)//60)))
    t2=time.perf_counter()
    D, I = index.search(query, k)
    t3=time.perf_counter()
    print("NSG(R=%d) Search:%dm%ds"%(R,(t3-t2)/60,t3-t2-60*((t3-t2)//60)))
    return D,I

def composite_indexes(X, index_string, dim = 100, k = 200, nprobe = 5, efConstruction=200, efSearch = 250):
    print("----------------------------------------------------------------")
    index = faiss.index_factory(dim, index_string)
    t0=time.perf_counter()
    index.train(X)
    if "HNSW" in index_string:
        faiss.ParameterSpace().set_index_parameter(index, "efConstruction", efConstruction)
    index.add(X)
    t1=time.perf_counter()
    if "HNSW" in index_string:
        print("composite_indexes(\"%s\",nprobe=%d,efConstruction=%d,efSearch=%d) Build:%dm%ds"%(index_string,nprobe,efConstruction,efSearch,(t1-t0)/60,t1-t0-60*((t1-t0)//60)))
    else:
        print("composite_indexes(\"%s\",nprobe=%d) Build:%dm%ds"%(index_string,nprobe,(t1-t0)/60,t1-t0-60*((t1-t0)//60)))
    t2=time.perf_counter()
    if "HNSW" in index_string:
        faiss.ParameterSpace().set_index_parameter(index, "efSearch", efSearch)
    if "IVF" in index_string:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)
    if "IMI" in index_string:
        imi = faiss.extract_index_ivf(index)  # access nprobe
        imi.nprobe = nprobe
    D,I=index.search(X, k)
    t3=time.perf_counter()
    if "HNSW" in index_string:
        print("composite_indexes(\"%s\",nprobe=%d,efConstruction=%d,efSearch=%d) Search %d-NN:%dm%ds"%(index_string,nprobe,efConstruction,efSearch,k,(t3-t2)/60,t3-t2-60*((t3-t2)//60)))
    else:
        print("composite_indexes(\"%s\",nprobe=%d) Search %d-NN:%dm%ds"%(index_string,nprobe,k,(t3-t2)/60,t3-t2-60*((t3-t2)//60)))
    del index
    return D,I

