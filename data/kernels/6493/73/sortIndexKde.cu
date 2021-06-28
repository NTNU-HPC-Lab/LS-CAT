#include "includes.h"
__global__ void sortIndexKde ( const int d, const int n, const float *a, const float *b, float *sa, float *sb ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;
int ij = i + j * d;
int mewj, il;
float mewa, mewb;
if ( i < d && j < n ) {
mewj = j;
mewa = a[ij];
mewb = b[ij];
for ( int l = 0; l < n; l++ ) {
il = i + l * d;
if ( l > j ) {
mewj += ( a[il] > mewa ) * ( l - mewj );
mewa += ( a[il] > mewa ) * ( a[il] - mewa );
mewb += ( a[il] > mewa ) * ( b[il] - mewb );
} else if ( l < j ) {
mewj += ( a[il] < mewa ) * ( l - mewj );
mewa += ( a[il] < mewa ) * ( a[il] - mewa );
mewb += ( a[il] < mewa ) * ( b[il] - mewb );
}
}
sa[ij] = mewa;
sb[ij] = mewb;
}
}