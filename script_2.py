import argparse
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

def main(n,m,l,d):
    srng = RandomStreams(seed=234)
    #what does seed number mean?
    rv_a = srng.normal((n,m), avg = 0, std = 1)
    rv_b = srng.normal((n,l), avg = 0, std = 1)
    rv_x = theano.shared(numpy.random.uniform( size=(m, l), low=d, high=d)) 
    f_a = theano.function([], rv_a, no_default_updates=False)
    #unless seed number changed, it always give the same output, no randomness?
    f_b = theano.function([], rv_b)
    f_x = theano.function([], rv_x)

    A = T.dmatrix('A')
    B = T.dmatrix('B')
    X = T.dmatrix('X')
    Y = T.dot(A, X) + B
    f = theano.function([A, B, X], Y)
    print f(f_a(), f_b(), f_x())

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
    main(n,m,l,d)
