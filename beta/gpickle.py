import os
import sys
import pickle
import getopt
import logging
import numpy as np
import networkx as nx


def read_grp(f, base_dir=None):
    """Read list of graph file names."""
    if not base_dir is None:
        return [os.path.join(base_dir, x.strip()) for x in open(f)]
    return [x.strip() for x in open(f)]


def load_grp(L):
    """Build networkx graphs."""    
    graphs = []
    for x in L:
        A = np.genfromtxt(x)
        A[np.where(A >= 1)] = 1
        G = nx.Graph(A)
        
        label_file = "%s.vertexLabel" % x
        if os.path.exists(label_file):
            labels = np.genfromtxt(label_file)

            assert len(labels) == len(G)
            for idx,l in enumerate(labels):
                G.node[idx]['type'] = int(l)
        graphs.append(G)
    return graphs


def usage():
    print """
Usage: python gpickle.py [OPTIONS]
    
    OPTIONS (required):
        -l [arg] - List with graph file names
        -b [arg] - Base directory
        -o [arg] - Output file
        -c [arg] - Label file
"""


def main(argv=None):
    if argv is None: argv = sys.argv

    opts, args = getopt.getopt(argv[1:], "hb:l:o:c:")

    G_file = None # (G)graph files
    G_base = None # (G)graph base directory
    D_dump = None # (D)data dump file (pickled)
    C_file = None # (C)lass labels

    for opt, arg in opts:
        if opt == "-h":
            usage()
            sys.exit(os.EX_OK)
        if opt == "-l":
            G_file = arg
        if opt == "-b":
            G_base = arg
        if opt == "-o":
            D_dump = arg
        if opt == "-c":
            C_file = arg

    G_list = read_grp(G_file, G_base)
    G_data = load_grp(G_list)
    C_list = np.genfromtxt(C_file)
    
    D = {'G_dat' : G_data, 
         'G_cls' : C_list}
    if not D_dump is None:
        pickle.dump(D, open(D_dump, "w"))
        
        
if __name__ == "__main__":
    main()