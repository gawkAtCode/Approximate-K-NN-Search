# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 2 clause

import time

import numba
from numba.core import types
import numba.experimental.structref as structref
import numpy as np


@numba.njit("void(i8[:], i8)", cache=True)
def seed(rng_state, seed):
    """Seed the random number generator with a given seed."""
    rng_state.fill(seed + 0xFFFF)


@numba.njit("i4(i8[:])", cache=True)
def tau_rand_int(state):
    """A fast (pseudo)-random number generator.

    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    A (pseudo)-random int32 value
    """
    state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ (
        (((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ (
        (((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ (
        (((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11
    )

    return state[0] ^ state[1] ^ state[2]


@numba.njit("f4(i8[:])", cache=True)
def tau_rand(state):
    """A fast (pseudo)-random number generator for floats in the range [0,1]

    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    A (pseudo)-random float32 in the interval [0, 1]
    """
    integer = tau_rand_int(state)
    return abs(float(integer) / 0x7FFFFFFF)

@structref.register
class HeapType(types.StructRef):
    pass


class Heap(structref.StructRefProxy):
    @property
    def indices(self):
        return Heap_get_indices(self)

    @property
    def distances(self):
        return Heap_get_distances(self)


@numba.njit(cache=True)
def Heap_get_distances(self):
    return self.distances


@numba.njit(cache=True)
def Heap_get_indices(self):
    return self.indices


structref.define_proxy(Heap, HeapType, ["indices", "distances"])

# Heap = namedtuple("Heap", ("indices", "distances"))


@numba.njit(cache=True)
def make_heap(n_points, size):
    """Constructor for the numba enabled heap objects. The heaps are used
    for approximate nearest neighbor search, maintaining a list of potential
    neighbors sorted by their distance. We also flag if potential neighbors
    are newly added to the list or not. Internally this is stored as
    a single ndarray; the first axis determines whether we are looking at the
    array of candidate graph_indices, the array of distances, or the flag array for
    whether elements are new or not. Each of these arrays are of shape
    (``n_points``, ``size``)

    Parameters
    ----------
    n_points: int
        The number of graph_data points to track in the heap.

    size: int
        The number of items to keep on the heap for each graph_data point.

    Returns
    -------
    heap: An ndarray suitable for passing to other numba enabled heap functions.
    """
    indices = np.full((int(n_points), int(size)), -1, dtype=np.int32)
    distances = np.full((int(n_points), int(size)), np.infty, dtype=np.float32)
    result = (indices, distances)

    return result

@numba.njit(parallel=True, cache=False)
def deheap_sort(indices, distances):
    """Given two arrays representing a heap (indices and distances), reorder the
     arrays by increasing distance. This is effectively just the second half of
     heap sort (the first half not being required since we already have the
     graph_data in a heap).

     Note that this is done in-place.

    Parameters
    ----------
    indices : array of shape (n_samples, n_neighbors)
        The graph indices to sort by distance.
    distances : array of shape (n_samples, n_neighbors)
        The corresponding edge distance.

    Returns
    -------
    indices, distances: arrays of shape (n_samples, n_neighbors)
        The indices and distances sorted by increasing distance.
    """
    for i in numba.prange(indices.shape[0]):
        # starting from the end of the array and moving back
        for j in range(indices.shape[1] - 1, 0, -1):
            indices[i, 0], indices[i, j] = indices[i, j], indices[i, 0]
            distances[i, 0], distances[i, j] = distances[i, j], distances[i, 0]

            siftdown(distances[i, :j], indices[i, :j], 0)
    return indices, distances


@numba.njit(parallel=True, locals={"idx": numba.types.int32,"i": numba.types.int32,"d":numba.types.float32}, cache=True)
def new_build_candidates(current_graph, max_candidates, rng_state, n_threads):
    """Build a heap of candidate neighbors for nearest neighbor descent. For
    each vertex the candidate neighbors are any current neighbors, and any
    vertices that have the vertex as one of their nearest neighbors.

    Parameters
    ----------
    current_graph: heap
        The current state of the graph for nearest neighbor descent.

    max_candidates: int
        The maximum number of new candidate neighbors.

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    candidate_neighbors: A heap with an array of (randomly sorted) candidate
    neighbors for each vertex in the graph.
    """
    current_indices = current_graph[0]
    current_priority = current_graph[1]
    #current_flags = current_graph[2]

    n_vertices = current_indices.shape[0]
    n_neighbors = current_indices.shape[1]

    new_candidate_indices = np.full((n_vertices, max_candidates), -1, dtype=np.int32)
    new_candidate_priority = np.full(
        (n_vertices, max_candidates), np.inf, dtype=np.float32
    )
    old_candidate_indices=new_candidate_indices
    old_candidate_priority=new_candidate_priority
    for n in numba.prange(n_threads):
        local_rng_state = rng_state + n
        for i in range(n_vertices):
            if i % n_threads == n:
                for j in range(n_neighbors):
                    idx = current_indices[i, j]
                    d = tau_rand(local_rng_state)
                    checked_heap_push(
                        new_candidate_priority[i], new_candidate_indices[i], d, idx
                    )
    
    for n in numba.prange(n_threads):
        local_rng_state = rng_state + n
        for i in range(n_vertices):
            if i % n_threads == n:
                for j in range(n_neighbors):
                    idx = current_indices[i, j]
                    d = tau_rand(local_rng_state)
                    checked_heap_push(
                        new_candidate_priority[idx],
                        new_candidate_indices[idx],
                        d,
                        i,
                    )

    return new_candidate_indices, old_candidate_indices


@numba.njit(
    "i4(f4[::1],i4[::1],f4,i4)",
    fastmath=True,
    locals={
        "size": numba.types.intp,
        "i": numba.types.uint16,
        "ic1": numba.types.uint16,
        "ic2": numba.types.uint16,
        "i_swap": numba.types.uint16,
    },
    cache=True,
)
def checked_heap_push(priorities, indices, p, n):
    if p >= priorities[0]:
        return 0

    #pynnd third change
    #if n in indices: 
    #    return 0
    size = priorities.shape[0]

    # break if we already have this element.
    for i in range(size):
        if n == indices[i]:
            return 0



    # insert val at position zero
    priorities[0] = p
    indices[0] = n

    # descend the heap, swapping values until the max heap criterion is met
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= size:
            break
        elif ic2 >= size:
            if priorities[ic1] > p:
                i_swap = ic1
            else:
                break
        elif priorities[ic1] >= priorities[ic2]:
            if p < priorities[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if p < priorities[ic2]:
                i_swap = ic2
            else:
                break

        priorities[i] = priorities[i_swap]
        indices[i] = indices[i_swap]

        i = i_swap

    priorities[i] = p
    indices[i] = n

    return 1


@numba.njit(
    "i4(f4[::1],i4[::1],f4,i4,u1,i4[::1],f4[::1])",
    fastmath=True,
    locals={
        "size": numba.types.intp,
        "i": numba.types.uint16,
        "ic1": numba.types.uint16,
        "ic2": numba.types.uint16,
        "i_swap": numba.types.uint16,
    },
    cache=True,
)
def checked_flagged_heap_push(priorities, indices, p, n, f, init_graph, init_dist):
    if p >= priorities[0]:
        return 0
        
    size = priorities.shape[0]

    for i in range(size-1,0,-1):
        if n == indices[i]:
            return 0

    # insert val at position zero
    priorities[0] = p
    indices[0] = n
    #flags[0] = f

    # descend the heap, swapping values until the max heap criterion is met
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= size:
            break
        elif ic2 >= size:
            if priorities[ic1] > p:
                i_swap = ic1
            else:
                break
        elif priorities[ic1] >= priorities[ic2]:
            if p < priorities[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if p < priorities[ic2]:
                i_swap = ic2
            else:
                break

        priorities[i] = priorities[i_swap]
        indices[i] = indices[i_swap]
        #flags[i] = flags[i_swap]

        i = i_swap

    priorities[i] = p
    indices[i] = n
    #flags[i] = f

    return 1


@numba.njit(
    parallel=True,
    locals={
        "p": numba.int32,
        "q": numba.int32,
        "d": numba.float32,
        "added": numba.uint8,
        "n": numba.uint32,
        "i": numba.uint32,
        "j": numba.uint32,
    },
    cache=False,
)
def apply_graph_updates_low_memory(current_graph, updates, n_threads, init_graph, init_dist):

    n_changes = 0
    priorities = current_graph[1]
    indices = current_graph[0]

    for n in numba.prange(n_threads):
        for i in range(len(updates)):
            if i%n_threads == n:
                for j in range(len(updates[i])):
                    p, q, d = updates[i][j]

                    if p == -1 or q == -1:
                        continue

                    checked_flagged_heap_push(
                        priorities[p], indices[p], d, q, 1, init_graph[p], init_dist[p]
                    )

                    checked_flagged_heap_push(
                        priorities[q], indices[q], d, p, 1, init_graph[q], init_dist[q]
                    )

    return n_changes

@numba.njit(cache=True)
def initalize_heap_from_graph_indices(heap, graph_indices, data, metric):

    for i in range(graph_indices.shape[0]):
        for idx in range(graph_indices.shape[1]):
            j = graph_indices[i, idx]
            if j >= 0:
                d = metric(data[i], data[j])
                checked_flagged_heap_push(heap[1][i], heap[0][i], d, j, 1, None, None)
    return heap

@numba.njit(parallel=True, locals={"i": numba.types.int64},cache=False)
def update_from_reverse_neighbors(
    heap, graph_indices, graph_distances
):    
    n_threads = numba.get_num_threads()
    for n in numba.prange(n_threads):
        for i in range(graph_indices.shape[0]):
            if i % n_threads == n:
                for j in range(graph_indices.shape[1]):
                    idx = graph_indices[i, j]
                    checked_heap_push(
                        heap[1][idx],
                        heap[0][idx],
                        graph_distances[i,j],
                        i,
                    )
    return heap

@numba.njit(cache=True)
def initalize_heap_from_graph_indices_and_distances(
    heap, graph_indices, graph_distances
):
    for i in range(graph_indices.shape[0]):
        for idx in range(graph_indices.shape[1]):
            j = graph_indices[i, idx]
            d = graph_distances[i, idx]
            heap[1][i][graph_indices.shape[1]-idx-1]=d
            heap[0][i][graph_indices.shape[1]-idx-1]=j
            #heap[2][i][graph_indices.shape[1]-idx-1]=1
    return heap


@numba.njit(parallel=True, cache=False)
def sparse_initalize_heap_from_graph_indices(
    heap, graph_indices, data_indptr, data_indices, data_vals, metric
):

    for i in numba.prange(graph_indices.shape[0]):
        for idx in range(graph_indices.shape[1]):
            j = graph_indices[i, idx]
            ind1 = data_indices[data_indptr[i] : data_indptr[i + 1]]
            data1 = data_vals[data_indptr[i] : data_indptr[i + 1]]
            ind2 = data_indices[data_indptr[j] : data_indptr[j + 1]]
            data2 = data_vals[data_indptr[j] : data_indptr[j + 1]]
            d = metric(ind1, data1, ind2, data2)
            checked_flagged_heap_push(heap[1][i], heap[0][i], d, j, 1, None, None)

    return heap


# Generates a timestamp for use in logging messages when verbose=True
def ts():
    return time.ctime(time.time())
