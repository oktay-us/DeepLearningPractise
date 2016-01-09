import argparse
parser = argparse.ArgumentParser(description='number_from_user')
parser.add_argument('inp', type=float)
args = parser.parse_args()
print(args.inp * 10 + 8)

import theano.tensor as T
from theano import function
x = T.dscalar('x')
#y = T.dscalar('y') * Don't need this
# Type is not matching, namaspace type to theano tensorType
#x = args.inp ** you reassigned x here.
y = 10 * x + 8
f = function([x], y)

print f(args.inp)