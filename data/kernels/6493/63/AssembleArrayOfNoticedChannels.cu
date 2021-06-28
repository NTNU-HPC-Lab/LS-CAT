#include "includes.h"
__global__ void AssembleArrayOfNoticedChannels ( const int nmbrOfChnnls, const float lwrNtcdEnrg, const float hghrNtcdEnrg, const float *lwrChnnlBndrs, const float *hghrChnnlBndrs, const float *gdQltChnnls, float *ntcdChnnls ) {
int c = threadIdx.x + blockDim.x * blockIdx.x;
if ( c < nmbrOfChnnls ) {
ntcdChnnls[c] = ( lwrChnnlBndrs[c] > lwrNtcdEnrg ) * ( hghrChnnlBndrs[c] < hghrNtcdEnrg ) * ( 1 - gdQltChnnls[c] );
}
}