'''
Module to do x * 10 + 8 with Theano
'''

import argparse
import theano
import theano.tensor as T


def main(x):
    X = T.dscalar('X')
    Y = 10 * X + 8
    f = theano.function([X], Y)

    print f(x)

def get_parser():
    parser = argparse.ArgumentParser(description='number_from_user')
    parser.add_argument('x', type=float)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    x = args.x
    main(x)