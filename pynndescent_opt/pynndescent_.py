# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 2 clause

from warnings import warn
import sys
import numba
import time
from numba.np.ufunc import workqueue
import numpy as np
from sklearn.utils import check_random_state, check_array

from numba import jit, objmode
import datetime
import heapq

from .distances import *

from .utils import (
    tau_rand_int,
    tau_rand,
    make_heap,
    new_build_candidates,
    ts,
    checked_flagged_heap_push,
    apply_graph_updates_low_memory,
    initalize_heap_from_graph_indices,
    initalize_heap_from_graph_indices_and_distances,
    sparse_initalize_heap_from_graph_indices,
    update_from_reverse_neighbors,
    checked_heap_push,
)

update_type = numba.types.List(
    numba.types.List((numba.types.int64, numba.types.int64, numba.types.float64))
)

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

FLOAT32_EPS = np.finfo(np.float32).eps

EMPTY_GRAPH = make_heap(1, 1)

@numba.njit(parallel=True, cache=False)
def generate_graph_updates(
    new_candidate_block, dist_thresholds, data, dist
):

    block_size = new_candidate_block.shape[0]
    updates = [[(-1, -1, np.inf)] for i in range(block_size)]
    max_candidates = new_candidate_block.shape[1]

    n_threads = numba.get_num_threads()
    for n in numba.prange(n_threads):
        for i in range(block_size):
            if i % n_threads == n:
                for j in range(max_candidates):
                    p = new_candidate_block[i, j]
                    if p < 0:
                        continue

                    for k in range(j, max_candidates):
                        q = new_candidate_block[i, k]
                        if q < 0:
                            continue
                        if dist_thresholds[p] <= 0.95 and dist_thresholds[q] <=0.95:
                            continue
                        d = dist(data[p], data[q])
                        if d <= dist_thresholds[p] or d <= dist_thresholds[q]:
                            updates[i].append((p, q, d))
                    
    return updates


@numba.njit(cache=False)
def process_candidates(
    data,
    dist,
    current_graph,
    new_candidate_neighbors,
    n_blocks,
    block_size,
    n_threads,
    faiss_graph=None,
    faiss_dist=None
):
    n_vertices = new_candidate_neighbors.shape[0]
    for i in range(n_blocks + 1):
        block_start = i * block_size
        block_end = min(n_vertices, (i + 1) * block_size)

        new_candidate_block = new_candidate_neighbors[block_start:block_end]
 
        dist_thresholds = current_graph[1][:, 0]

        updates = generate_graph_updates(
            new_candidate_block, dist_thresholds, data, dist,
        )

        apply_graph_updates_low_memory(current_graph, updates, n_threads, faiss_graph, faiss_dist)


@numba.njit()
def nn_descent_internal_low_memory_parallel(
    current_graph,
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=euclidean,
    n_iters=10,
    delta=0.001,
    verbose=False,
    faiss_graph=None,
    faiss_dist=None
):
    n_vertices = data.shape[0]
    block_size = 65536
    n_blocks = n_vertices // block_size
    n_threads = numba.get_num_threads()

    for n in range(n_iters):
        if verbose:
            print("\t", n + 1, " / ", n_iters)
        (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph, max_candidates, rng_state, n_threads
        )

        process_candidates(
            data,
            dist,
            current_graph,
            new_candidate_neighbors,
            n_blocks,
            block_size,
            n_threads,
            faiss_graph, 
            faiss_dist,
        )

@numba.njit()
def nn_descent(
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=euclidean,
    n_iters=10,
    delta=0.001,
    low_memory=True,
    init_graph=EMPTY_GRAPH,
    verbose=False,
    faiss_graph=None, 
    faiss_dist=None
):
    if init_graph[0].shape[0] == 1:  # EMPTY_GRAPH
        current_graph = make_heap(data.shape[0], n_neighbors)
    elif (
        init_graph[0].shape[0] == data.shape[0]
        and init_graph[0].shape[1] == n_neighbors
    ):
        current_graph = init_graph
    else:
        raise ValueError("Invalid initial graph specified!")
    if low_memory:
        nn_descent_internal_low_memory_parallel(
            current_graph,
            data,
            n_neighbors,
            rng_state,
            max_candidates=max_candidates,
            dist=dist,
            n_iters=n_iters,
            delta=delta,
            verbose=verbose,
            faiss_graph=faiss_graph, 
            faiss_dist=faiss_dist
        )
    else:
        print("high_memory has not yet been implemented")
    return current_graph[0], current_graph[1]

class NNDescent:
    """NNDescent for fast approximate nearest neighbor queries. NNDescent is
    very flexible and supports a wide variety of distances, including
    non-metric distances. NNDescent also scales well against high dimensional
    graph_data in many cases. This implementation provides a straightfoward
    interface, with access to some tuning parameters.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The training graph_data set to find nearest neighbors in.

    metric: string or callable (optional, default='euclidean')
        The metric to use for computing nearest neighbors. If a callable is
        used it must be a numba njit compiled function. Supported metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
            * hellinger
            * wasserstein-1d
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.

    n_neighbors: int (optional, default=30)
        The number of neighbors to use in k-neighbor graph graph_data structure
        used for fast approximate nearest neighbor search. Larger values
        will result in more accurate search results at the cost of
        computation time.

    init_graph: np.ndarray (optional, default=None)
        2D array of indices of candidate neighbours of the shape
        (data.shape[0], n_neighbours). If the j-th neighbour of the i-th
        instances is unknown, use init_graph[i, j] = -1

    init_dist: np.ndarray (optional, default=None)
        2D array with the same shape as init_graph,
        such that metric(data[i], data[init_graph[i, j]]) equals
        init_dist[i, j]

    random_state: int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    algorithm: string (optional, default='standard')
        This implementation provides an alternative algorithm for
        construction of the k-neighbors graph used as a search index. The
        alternative algorithm can be fast for large ``n_neighbors`` values.
        The``'alternative'`` algorithm has been deprecated and is no longer
        available.

    low_memory: boolean (optional, default=True)
        Whether to use a lower memory, but more computationally expensive
        approach to index construction.

    max_candidates: int (optional, default=None)
        Internally each "self-join" keeps a maximum number of candidates (
        nearest neighbors and reverse nearest neighbors) to be considered.
        This value controls this aspect of the algorithm. Larger values will
        provide more accurate search results later, but potentially at
        non-negligible computation cost in building the index. Don't tweak
        this value unless you know what you're doing.

    n_iters: int (optional, default=None)
        The maximum number of NN-descent iterations to perform. The
        NN-descent algorithm can abort early if limited progress is being
        made, so this only controls the worst case. Don't tweak
        this value unless you know what you're doing. The default of None means
        a value will be chosen based on the size of the graph_data.

    delta: float (optional, default=0.001)
        Controls the early abort due to limited progress. Larger values
        will result in earlier aborts, providing less accurate indexes,
        and less accurate searching. Don't tweak this value unless you know
        what you're doing.

    n_jobs: int or None, optional (default=None)
        The number of parallel jobs to run for neighbors index construction.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    compressed: bool (optional, default=False)
        Whether to prune out data not needed for searching the index. This will
        result in a significantly smaller index, particularly useful for saving,
        but will remove information that might otherwise be useful.

    verbose: bool (optional, default=False)
        Whether to print status graph_data during the computation.
    """

    def __init__(
        self,
        data,
        metric="euclidean",
        metric_kwds=None,
        n_neighbors=30,
        init_graph=None,
        init_dist=None,
        random_state=None,
        low_memory=True,
        max_candidates=None,
        n_iters=None,
        delta=0.001,
        n_jobs=None,
        compressed=False,
        verbose=False,
    ):


        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.max_candidates = max_candidates
        self.low_memory = low_memory
        self.n_iters = n_iters
        self.delta = delta
        self.dim = data.shape[1]
        self.n_jobs = n_jobs
        self.compressed = compressed
        self.verbose = verbose

        data = check_array(data, dtype=np.float32, accept_sparse="csr", order="C")
        self._raw_data = data

        metric_kwds = metric_kwds or {}
        self._dist_args = tuple(metric_kwds.values())

        self.random_state = random_state

        current_random_state = check_random_state(self.random_state)

        self._distance_correction = None

        if callable(metric):
            _distance_func = metric
        elif metric in named_distances:
            if metric in fast_distance_alternatives:
                _distance_func = fast_distance_alternatives[metric]["dist"]
                self._distance_correction = fast_distance_alternatives[
                    metric
                ]["correction"]
            else:
                _distance_func = named_distances[metric]
        else:
            raise ValueError("Metric is neither callable, " + "nor a recognised string")

        # Create a partial function for distances with arguments
        if len(self._dist_args) > 0:
            dist_args = self._dist_args

            @numba.njit()
            def _partial_dist_func(x, y):
                return _distance_func(x, y, *dist_args)

            self._distance_func = _partial_dist_func
        else:
            self._distance_func = _distance_func

        if metric in (
            "cosine",
            "dot",
            "correlation",
            "dice",
            "jaccard",
            "hellinger",
            "hamming",
        ):
            self._angular_trees = True
        else:
            self._angular_trees = False

        self.rng_state = current_random_state.randint(INT32_MIN, INT32_MAX, 3).astype(
            np.int64
        )
         
        self._original_num_threads = numba.get_num_threads()

        if self.n_jobs != -1 and self.n_jobs is not None:
            numba.set_num_threads(self.n_jobs)

        self._is_sparse = False
        
        if init_graph is None:
            _init_graph = EMPTY_GRAPH
        else:
            _init_graph = make_heap(init_graph.shape[0], self.n_neighbors)
            _init_graph = initalize_heap_from_graph_indices_and_distances(
                _init_graph, init_graph[:,:_init_graph[1].shape[1]], init_dist[:,:_init_graph[1].shape[1]]
            )
            _init_graph = update_from_reverse_neighbors(
                _init_graph, init_graph, init_dist
            )
                            
        if verbose:
            print(ts(), "NN descent for", str(n_iters), "iterations")

        self._neighbor_graph = nn_descent(
            self._raw_data,
            self.n_neighbors,
            self.rng_state,
            self.max_candidates,
            self._distance_func,
            self.n_iters,
            self.delta,
            low_memory=self.low_memory,
            init_graph=_init_graph,
            verbose=verbose,
            faiss_graph=init_graph,
            faiss_dist=init_dist,
        )
            

        if np.any(self._neighbor_graph[0] < 0):
            warn(
                "Failed to correctly find n_neighbors for some samples."
                " Results may be less than ideal. Try re-running with"
                " different parameters."
            )

        numba.set_num_threads(self._original_num_threads)