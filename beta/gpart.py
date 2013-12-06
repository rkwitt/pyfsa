import networkx as nx
from networkx.algorithms import bipartite
from collections import defaultdict
import matplotlib.pyplot as plt
import random as rnd

from collections import deque

import numpy as np
import time
import sys
import os

nV = int(sys.argv[1]) # nr of vertices
nE = int(sys.argv[2]) # nr. of edges
nSeed = int(sys.argv[3]) # nr. of seed vertices
max_visit = int(sys.argv[4]) # max. vertex visits
max_level = int(sys.argv[5]) # max. cell radius


def bfs(G, seed_sample, max_visit=1, max_level=None):
    """Breadth-first search.
    """
    
    N = len(G.nodes())
    Q = deque(maxlen=N) 
    V = [0 for i in range(N)] # visit counter
    
    # C[i] is the set of vertices that form the cell of seed_sample[i]
    C = [[] for i in seed_sample]
    set_C = [set() for i in seed_sample]
    # L[i][j] stores the distance of the C[i][j] to seed_sample[i]
    L = [[] for i in seed_sample] 
        
        
    if max_level is None:
        max_level = N
    
   
    def get_neighborhood(v):
        return filter(lambda u: V[u]<max_visit, G.neighbors(v))
        
        
    for i,v in enumerate(seed_sample):
        Q.append((v, 0, i))
        V[v] += 1
        
    while len(Q)>0:
        v, dist, seed = Q.popleft()
        assert(dist <= max_level)
        
        if v in set_C:
            continue
        set_C.insert(v)
            
        C[seed].append(v)
        L[seed].append(dist)
        
        if dist < max_level:
            for u in get_neighborhood(v):
                Q.append((u, dist+1, seed))
                V[u] += 1
                
            
    return (C,L)

G = nx.gnm_random_graph(nV, nE, seed=1234)

#fig = plt.figure(figsize=(10,10))
#nx.draw_shell(G)
#plt.savefig("/Users/rkwitt/Desktop/graph.pdf")

rnd.seed(1234)
seed_sample = rnd.sample(G.nodes(), nSeed)
C, L = bfs(G, seed_sample, max_visit, max_level)

print seed_sample
for s in range(len(seed_sample)):
    print "seed: %d" % seed_sample[s]
    print C[s]
    print L[s]
    
    #if S[v] is None:
    #    print "%d: no seed (/no level)!" % v
    #else:
    #    print "%d: S=%d, L=%d" % (v, S[v], L[v])

#for s in seed_sample:
#    cell = filter(lambda u: S[u] == s, G.nodes())
#    #print nx.adjacency_matrix(G.subgraph(cell))
    
    
    
    
    
    












#for s in G.nodes():
    

