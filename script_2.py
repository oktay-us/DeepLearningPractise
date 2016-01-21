import argparse
import numpy as np
import random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


floatX = 'float32'

def main(n, m, l, d):
    srng = RandomStreams(seed=random.randint(0, 1000))
    #what does seed number mean?

    X = T.matrix('X', dtype=floatX)
    M = T.scalar('M', dtype='int64')
    N = T.scalar('N', dtype='int64')
    L = T.scalar('L', dtype='int64')

    rv_a = srng.normal((N, M), avg=0, std=1).astype(floatX)
    rv_b = srng.normal((N, L), avg=0, std=1).astype(floatX)

    Y = T.dot(rv_a, X) + rv_b
    f = theano.function([X, M, N, L], Y)

    x = np.zeros((m, l)).astype(floatX) + d

    print f(x, m, n, l)

def get_parser():
    parser = argparse.ArgumentParser(description='number_from_user')
    parser.add_argument('N', type=int)
    parser.add_argument('M', type=int)
    parser.add_argument('L', type=int)
    parser.add_argument('D', type=float)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    n = args.N
    m = args.M
    l = args.L
    d = args.D
    main(n, m, l, d)
