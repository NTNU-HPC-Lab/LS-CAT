#include "includes.h"
__global__ void copy_sort_int( const int *orig, const unsigned int *sort_idx, const unsigned int nitems, int *sorted ) {
for( int i = 0; i < nitems; ++ i ) {
sorted[sort_idx[i]] = orig[i];
}
}