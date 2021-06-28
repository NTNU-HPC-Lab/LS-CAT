#include "includes.h"
__global__ void modify_i_j( int width, int height, int pitch, float *d_array, int i, int j, float change_to ){
//we want to change the [i,j]-th of the 2-dim array
int idx = blockIdx.x; //row
int idy = threadIdx.x; //column

//we can do index by pointer:
//if ((idx == i) && (idy == j)){
//float* row = (float *)((char*)d_array + idx*pitch);
//	row[idy] = change_to;
//}

//or, a more convenient way is to do index just use idx and idy
if ((idx==i)&&(idy==j))
{
d_array[idx*(pitch/sizeof(float)) + idy] = change_to;
}

}