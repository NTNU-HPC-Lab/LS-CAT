#include "includes.h"
__global__ void sortIndex ( const int d, const int n, const float *a, int *si, float *sa ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;
int ij = i + j * d;
int mewj, il;
float mewa;
if ( i < d && j < n ) {
mewj = j;
mewa = a[ij];
for ( int l = 0; l < n; l++ ) {
il = i + l * d;
if ( l > j ) {
mewj += ( a[il] < mewa ) * ( l - mewj );
mewa += ( a[il] < mewa ) * ( a[il] - mewa );
} else if ( l < j ) {
mewj += ( a[il] > mewa ) * ( l - mewj );
mewa += ( a[il] > mewa ) * ( a[il] - mewa );
}
}
si[ij] = mewj;
sa[ij] = mewa;
}
}