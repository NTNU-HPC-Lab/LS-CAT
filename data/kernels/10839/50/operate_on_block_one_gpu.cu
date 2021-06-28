#include "includes.h"
__global__ void operate_on_block_one_gpu (int i, int *d_array, int x_start, int y_start, char *d_subsequence1, char *d_subsequence2)
{
long tid=blockIdx.x *blockDim.x + threadIdx.x;


int x_my_block = x_start - (blockIdx.x) * tileLength;
int y_my_block = y_start + (blockIdx.x) * tileLength;
//printf("%d%d%d%d\n", i,blockIdx.x + 1, x_my_block,y_my_block);


//operate_on_block (x_start, y_start, subsequence1, subsequence2);



int x_start_local = x_my_block;
int y_start_local = y_my_block;

for (int i = 1; i <= tileLength; ++i)
{
//#pragma omp parallel for
int j= (tid % tileLength) + 1;
//printf("%d-%d\n",j,i);
if (j <= i)
{
//printf("%d\n",j);
int x_my = x_start_local - (j-1)*1;
int y_my = y_start_local + (j-1)*1;
//printf("%d%d\n",x_my,y_my );
if (d_subsequence1 [x_my] == d_subsequence2 [y_my])
{
d_Z = 1 + d_B;
}
else
{
( d_A > d_C ? d_Z = d_A : d_Z = d_C );
}
}
x_start_local = x_start_local + 1;
}

x_start_local = x_start_local - 1;
y_start_local = y_start_local + 1;

for (int i = tileLength-1; i >= 1 ; --i)
{
int j= (tid % tileLength) + 1;
if (j <= i)
{
int x_my = x_start_local - (j-1)*1;
int y_my = y_start_local + (j-1)*1;
//printf("%d%d\n",x_my,y_my );
if (d_subsequence1 [x_my] == d_subsequence2 [y_my])
{
d_Z = 1 + d_B;
}
else
{
( d_A > d_C ? d_Z = d_A : d_Z = d_C );
}

}

y_start_local = y_start_local + 1;
}

}