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
max_level = int(sys.argv[5]) # max. cell radius
max_visit = int(sys.argv[4]) # max. vertex visits


def bfs(G, seed_sample, max_visit=1, max_level=None):
    """Breadth-first search.
    """
    print "Max level: %d" % max_level
    raw_input()
    
    N = len(G.nodes())
    Q = deque(maxlen=N) 
    V = [0 for i in range(N)] # visit counter
    S = [None for i in range(N)] # S[i] = seed vertex of i-th vertex
    L = [None for i in range(N)] # L[i] = distance of i-th vertex to S[i]
    
    if max_level is None:
        max_level = N
    
    def enqueue(v, prev):
        if not prev is None and L[prev]>=max_level:
            return
        
        Q.append(v)
        V[v] += 1 # INC visit count for v-th vertex
        if not prev is None:
            S[v] = S[prev] # set seed for v-th vertex
            L[v] = L[prev]+1 # INC distance to seed by 1
    
    def get_neighborhood(v):
        return filter(lambda u: V[u]<max_visit, G.neighbors(v))
        
    for v in seed_sample:
        S[v] = v
        L[v] = 0
        enqueue(v, None)
        
    while len(Q)>0:
        v = Q.popleft()
        assert(L[v] <= max_level)
        
        for u in get_neighborhood(v):
            enqueue(u,v)
       
    return (S,L)

G = nx.gnm_random_graph(nV, nE, seed=1234)

#fig = plt.figure(figsize=(10,10))
#nx.draw_shell(G)
#plt.savefig("/Users/rkwitt/Desktop/graph.pdf")

rnd.seed(1234)
seed_sample = rnd.sample(G.nodes(), nSeed)
S, L = bfs(G, seed_sample, max_visit, max_level)

print seed_sample
for v in range(len(S)):
    if S[v] is None:
        print "%d: no seed (/no level)!" % v
    else:
        print "%d: S=%d, L=%d" % (v, S[v], L[v])

for s in seed_sample:
    cell = filter(lambda u: S[u] == s, G.nodes())
    #print nx.adjacency_matrix(G.subgraph(cell))
    
    
    
    
    
    












#for s in G.nodes():
    

