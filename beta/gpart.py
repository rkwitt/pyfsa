"""GFSA: Generalized fine-strucure analysis
"""


from collections import defaultdict
import matplotlib.pyplot as plt
import random as rnd

from collections import deque

import networkx as nx
import numpy as np
import getopt
import pickle
import time
import sys
import os


def usage():
    pass    


def load(l, b=None):
    """Load pickled graphs.
    """
    if not b is None:
        L = [os.path.join(b, x.strip()) for x in open(l)]
    else:
        L = [x.split() for x in open(l)]
    
    data = []
    for e in L:
        data.append(pickle.load(e))
    return data
    

def main(argv=None):

    num_vseeds = None # N seed vertices
    multiscale = None # Build multiscale features
    graph_list = None # List of filenames with pickled graphs
    
    opts, args = getopt.getopt(sys.argv[1:], "hmv:l:s:")

    for opt, arg in opts:
        if opt == "-h":
            usage()
            sys.exit(os.EX_OK)
        if opt == "-v":
            max_visit = int(arg)
        if opt == "-l":
            max_level = int(arg)
        if opt == "-s":
            num_seeds = int(arg)
        if opt == "-m":
            multiscale = True
        if opt == "-i"      
        
    
    if graph_list is None:
        print "OOps: no graph list given ..."
        sys.exit(os.EX_OK)
    
    t0 = time.clock()
    data = load(graph_list)
    t1 = time.clock()
    
    t_bfs = 0 # cummulative runtime for BFS
    t_fun = 0 # cummulative runtime for attribute computation
                
    for (G,l) in data:
        V = G.nodes()
        
        seed_sample = range(V)
        if not n_seeds is None: 
            seed_sample = rnd.sample(V, n_seed)
       
        # run BFS
        t0 = time.clock()
        (C,L) = bfs(G, seed_sample, max_visit, max_level)
        t1 = time.clock()
        t_bfs += (t1-t0)
        
        if multiscale:
            F_tmp = 
        else:    
            F_tmp = np.zeros((len(C),len(funs))) 
         
        t0 =  time.clock()
        for i,s in C:
            sg_nodes = C[s] # partition nodes
            sg_dists = L[s] # distance to seed vertices
            
            if multiscale:
                
                # TODO: slice out the cell subgraphs 
                # for each distance!
                
                
            else:
                sg = G.subgraph(sg_nodes)
                if sg.nodes() == 1:
                    continue
                v = [f(sg) for f in funs]
                F_tmp[i,:] = np.asarray(v)

        t1 = time.clock()
        t_fun += (t1-t0)
       
     
    # outputs some runtime stats ...
    print "time (bfs/graph): %.3f" : float(t_bfs)/len(data) 
    print "time (fun/graph): %.3f" : float(t_fun)/len(data)
      
    # saves data to disk for further use
    np.savetxt(dat, dat_name_out)
    np.savetxt(idx, idx_name_out)
    np.savetxt(lab, lab_name_out)
            
     
    
    
    
    #nV = int(sys.argv[1]) # nr of vertices
    #nE = int(sys.argv[2]) # nr. of edges
    #nSeed = int(sys.argv[3]) # nr. of seed vertices
    #max_visit = int(sys.argv[4]) # max. vertex visits
    #max_level = int(sys.argv[5]) # max. cell radius
    
    
    



def bfs(G, seed_sample, max_visit=1, max_level=None):
    """Breadth-first search.
    """
    
    N = len(G.nodes())
    Q = deque() 
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
       
        if v in set_C[seed]:
            continue
        set_C[seed].add(v)
            
        C[seed].append(v)
        L[seed].append(dist)
        
        if dist < max_level:
            for u in get_neighborhood(v):
                Q.append((u, dist+1, seed))
                V[u] += 1
            
    return (C,L)


if __name__ == "__main__":
    main()



#G = nx.gnm_random_graph(nV, nE, seed=1234)

#fig = plt.figure(figsize=(10,10))
#nx.draw_shell(G)
#plt.savefig("/Users/rkwitt/Desktop/graph.pdf")

#t0 = time.clock()

#rnd.seed(1234)
#seed_sample = rnd.sample(G.nodes(), nSeed)
#C, L = bfs(G, seed_sample, max_visit, max_level)

#t1 = time.clock()
#print "%.3g [sec]" % (t1-t0)

#for s in range(len(seed_sample)):
#    print "seed: %d" % seed_sample[s]
#    print C[s]
#    print L[s]
    
 
