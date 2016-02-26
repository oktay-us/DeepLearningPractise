import argparse
import numpy as np
import random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


floatX = 'float32'

def main(n, l, n_it, d):
    srng = RandomStreams(seed=random.randint(0, 1000))
    X = T.matrix('X', dtype=floatX)
    N_IT = T.scalar('N_IT', dtype='int64')
    N = T.scalar('N', dtype='int64')
    L = T.scalar('L', dtype='int64')
    rv_a = srng.normal((N, N), avg=0, std=1).astype(floatX)
    rv_b = srng.normal((N, L), avg=0, std=1).astype(floatX)
    #import pdb; pdb.set_trace()
    Y = X
    for i in range(n_it):
           Y = T.dot(rv_a, Y) + rv_b

    f = theano.function([X, N, L], Y)
 
    x = np.zeros((n, l)).astype(floatX) + d
    y = f(x, n, l)
    print y
    
def get_parser():
    parser = argparse.ArgumentParser(description='number_from_user')
    parser.add_argument('N', type=int)
    parser.add_argument('L', type=int)
    parser.add_argument('N_IT', type=int)
    parser.add_argument('D', type=float)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    n = args.N
    l = args.L
    n_it = args.N_IT
    d = args.D
    main(n, l, n_it, d)
