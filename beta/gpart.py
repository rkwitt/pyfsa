"""GFSA: Generalized Fine-Structure Analysis
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Roland Kwitt, University of Salzburg; Stefan Huber, IST Austria, 2013"
__email__   = "E-Mail: roland.kwitt@kitware.com"
__status__  = "Development"


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


funs = [# Average degree
        lambda g : np.mean([e for e in g.degree().values()]),
        # Percentage of isolated points (i.e., degree(v) = 1)
        lambda g : float(len(np.where(np.array(nx.degree(g).values())==1)[0]))/g.order(),
        # Label entropy, as defined in [2]
        lambda g : label_entropy([e[1]['type'] for e in g.nodes(data=True)]),
        # Mixing coefficient of attributes
        #lambda g : np.linalg.det(nx.attribute_mixing_matrix(g,'type')),
        # Link impurity, as defined in [2]
        lambda g : link_impurity(g)]
     
            
def link_impurity(g):
    """Compute link impurity of vertex-labeled graph."""
    if len(g.nodes()) == 1:
        return 0
    edges = g.edges()
    u = np.array([g.node[a]['type'] for (a,b) in edges])
    v = np.array([g.node[b]['type'] for (a,b) in edges])
    return float(len(np.nonzero(u - v)[0]))/len(edges)


def label_entropy(labels):
    """Compute entropy of label vector."""
    H = np.bincount(labels)
    p = H[np.nonzero(H)].astype(float)/np.sum(H)
    return np.abs(-np.sum(p * np.log(p)))
    
    
def usage():
    print """
Usage: python gpart.py
"""
    sys.exit(os.EX_OK)
    

def main(argv=None):
    if argv is None: 
        argv = sys.argv
    
    opts, args = getopt.getopt(argv[1:], "hgv:l:s:i:m:o:")
    
    arg_out_file = None     # base name for output file (with features)
    arg_data_file = None    # input file with pickled graph data
    arg_max_visit = -1      # max. visit / vertex when running BFS
    arg_max_level = +1      # max. level for BFS
    arg_pbp_seeds = -1      # probability for choosing a vertex as a seed vertex
    arg_num_scale = []      # compute FSF for each level
    arg_run_global = False  # run global feature computation

    for opt, arg in opts:
        if opt == "-h":
            usage()
        if opt == "-i":
            arg_data_file = arg
        if opt == "-o":
            arg_out_file = arg
        if opt == '-g':
            arg_run_global = True
        if opt == "-v":
            arg_max_visit = int(arg)
        if opt == "-l":
            arg_max_level = int(arg)
        if opt == "-s":
            arg_pbp_seeds = float(arg)
        if opt == "-m":
            arg_num_scale = [int(x) for x in arg.split(',')]
    
    # Sanity checks ...
    if arg_data_file is None:
        print "OOps: data file not given given!"
        sys.exit()
    if not os.path.exists(arg_data_file):
        print "Oops: data file not existent!"
        sys.exit()
    
    # Data loading ...
    t0 = time.clock()
    data = pickle.load(open(arg_data_file, "r"))
    t1 = time.clock()
    print "loaded data in %.3g [sec]" % (t1-t0)

    graphs = data["G_dat"] # List of N networkx graphs
    labels = data["G_cls"] # List of N numeric class assignments
    
    # Initialize timings
    timings = dict()
    timings["bfs_comp"] = 0
    timings["fun_comp"] = 0
    
    statistics = []
    
    F = None # Feature matrix
    I = None # Index matrix
    
    # When running a global feature extraction, we do NOT need
    # indices; just the feature matrix which is (#graphs x #funs)
    if arg_run_global:
        F = np.zeros((len(graphs),len(funs)))
        I = np.zeros((len(graphs)))
    
    # Run over all graphs 
    for i,G in enumerate(graphs):
        print "graph %d" % i
        
        V = G.nodes()   # node list for G
        N = len(V)      # nr. of nodes in G
        
        if arg_run_global:
            F[i,:] = np.asarray([f(G) for f in funs])
            continue
        
        # Make sure that the probability is in [0,1]; if negative,
        # we want to select ALL nodes as seed nodes!
        if arg_pbp_seeds <= 0: 
            seed_sample = range(N)
        else:
            assert arg_pbp_seeds > 0 and arg_pbp_seeds < 1
            sel = int(np.round(float(N)*arg_pbp_seeds))
            sel = np.max((sel,1))
            seed_sample = rnd.sample(V, sel)
        
        max_visit = arg_max_visit
        max_level = arg_max_level
        if arg_max_visit <= 0: max_visit = N # i.e., no limit
        if arg_max_level <= 0: max_level = 1 # i.e., only subgraphs induced by nb. of radius 1
        
        # Run graph partitioning
        t0 = time.clock()
        (C,L) = bfs(G, seed_sample, max_visit, max_level)
        t1 = time.clock()
        timings["bfs_comp"] += (t1-t0)
    
        S = 1 # scales
        if len(arg_num_scale) > 1:
            S = len(arg_num_scale)
        V = np.zeros((len(seed_sample),len(funs)*S))
        
        # Iterate over all seed samples, get neighborhood-induced subgraphs
        # and run feature computation
        t0 = time.clock()
        G_stat = [0 for x in range(len(arg_num_scale))] 
        
        #print seed_sample
        degenerates = [] # record degenerate, i.e., V=1, cases
        
        for s in range(len(seed_sample)):
            #print i, s, C[s], L[s], len(G.nodes())
            #raw_input()
            
            sg_nodes = C[s] # nodes in cell induced by seed node s 
            sg_dists = L[s] # distances of nodes in cell to seed node s
            
            # When there are no scales given, we compute one feature
            # vector for the WHOLE neighborhood cell
            if not len(arg_num_scale):
                sg = G.subgraph(sg_nodes)
                if len(sg.nodes()) == 1:
                    degenerates.append(s)
                    print "skip %d" % s
                    continue
                V[s,:] = np.asarray([f(sg) for f in funs])
                #print s,V[s,:]
            # otherwise, we compute one feature vector for each level
            # up to max_level -> multiscale
            else:
                for j,r in enumerate(arg_num_scale):
                    assert r <= max_level 
                    
                    idx = filter(lambda u: sg_dists[u]<=r, range(len(sg_nodes)))
                    sg_r = G.subgraph([sg_nodes[x] for x in idx])
                    
                    # G_stat[j] records the total number of nodes in each of 
                    # the subgraphs of radius r = arg_num_scale[j] - This has 
                    # to be later divided by the number of seed nodes to get the 
                    # average number of nodes in subgraphs of radius r!
                    G_stat[j] += len(sg_r.nodes())
                   
                    # print "%d: %d/%d" % (r,len(sg_r.nodes()),len(G.nodes()))
                    # raw_input()
                    
                    if len(sg_r.nodes()) == 1:
                        degenerates.append(s)
                        continue
                    fs_r = np.asarray([f(sg_r) for f in funs])
                    V[s,j*len(funs):(j+1)*len(funs)] = fs_r
        
        # record the statistics for the current graph
        statistics.append( {'stat' : G_stat, 'nV' : len(G.nodes()), 'nE' : len(G.edges()) })
          
        # prune degenerates
        if len(degenerates):
            print "prune %d degenerate(s) ..."  % len(degenerates)
            V = np.delete(V, degenerates, axis=0)

        t1 = time.clock()
        timings["fun_comp"] += (t1-t0)
        
        # stack features
        if F is None: F = V
        else: 
            F = np.vstack((F,V))
        
        # stack indices
        if I is None: I = np.ones((V.shape[0],))
        else: 
            I = np.hstack((I,np.ones(V.shape[0],)*(i+1)))
    
    # output some timing statistics
    for k in timings.keys():
        print "time(%s/graph): %.10f / total=%.5f" % (k, timings[k]/len(graphs), timings[k])
    
    # output subgraph statistics: we have N graphs, R scales. The
    # S[i,j]-th entry contains the average number of nodes in the
    # subgraphs of radius arg_num_scale[j], extracted from the i-th
    # graph.
    if not arg_run_global:
        S = np.zeros((len(statistics),len(arg_num_scale)))
    
        for i,G_stat in enumerate(statistics):
            for j,x in enumerate(G_stat['stat']):
                S[i,j] = x/float(G_stat['nV']*len(seed_sample))
        for i,x in enumerate(arg_num_scale):
            print np.sum(S[:,i],axis=0)/len(statistics)
        
        
    # write feature matrix and the assignment of each feature vector
    # to a graph to HDD
    if not arg_out_file is None:
        f_mat_name = "%s.mat" % arg_out_file
        f_idx_name = "%s.idx" % arg_out_file
        
        np.savetxt(f_mat_name, F, delimiter=' ')
        np.savetxt(f_idx_name, I, delimiter=' ', fmt="%d")
            
 
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




    
 
