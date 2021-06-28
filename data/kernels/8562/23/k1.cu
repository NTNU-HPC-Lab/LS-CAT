#include "includes.h"
__global__ void k1( float* g_dataA, float* g_dataB, int floatpitch, int width)
{
extern __shared__ float s_data[];
// TODO, implement this kernel below
unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
y = y + 1; //because the edge of the data is not processed
// global thread(data) column index
unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
x = x + 1; //because the edge of the data is not processed
if( y >=width - 1|| x >= width - 1 || y < 1 || x < 1 )// this is to check to make sure that the thread is within the array.
return;

int startRow = blockIdx.y;
int startCol = blockDim.x * blockIdx.x;

int s_rowwidth = blockDim.x +2; // because the blocks have to overlap on the right side that is why you add 2
int s_index0 = threadIdx.x +1; //row zero in s_data. you add one because you don't deal with the outer edge
int s_index1 = threadIdx.x + s_rowwidth + 1; //row one in s_data.so this goes to the other side
int s_index2 = threadIdx.x + 2 * s_rowwidth +1; //this is to get the last
//int s_index_result = threadIdx.x + 3 * s_rowwidth + 1;
int mid_row = blockIdx.x * blockDim.x + 1 + floatpitch * blockIdx.y;

int g_index0 = (mid_row -1) * floatpitch + startCol + 1+ threadIdx.x;
int g_index1 = (mid_row) * floatpitch + startCol  + 1 + threadIdx.x;
int g_index2 = (mid_row +1) * floatpitch +startCol + 1 + threadIdx.x;

if(startCol + startRow + 1 < width -1)
{
//copy the data from gobal mem to shared mem
s_data[s_index0] = g_dataA[g_index0];
s_data[s_index1] = g_dataA[g_index1];
s_data[s_index2] = g_dataA[g_index2];

}//end of if statement to populate the middle row of the current block
if(startRow == 0)
{
//copy the extra two columns in the globabl mem
s_data[s_index0 -1] = g_dataA[g_index0 - 1];
s_data[s_index1 -1] = g_dataA[g_index1 -1];
s_data[s_index2 -1] = g_dataA[g_index2 -1];
}//end of if statement to populate the edge row
if(threadIdx.x == width -3 - startCol || threadIdx.x == blockDim.x-1)
{
s_data[s_index0 + 1] = g_dataA[g_index0 +1];
s_data[s_index1 + 1] = g_dataA[g_index1 +1];
s_data[s_index2 +1] = g_dataA[g_index2 + 1];
}//end of if statement to populate the row below the middle row

__syncthreads();

//if( x >= width - 1|| y >= width - 1 || x < 1 || y < 1 )// this is to check to make sure that the thread is within the array.
//	return;

//this is copied from the other kernel
g_dataB[y * width + x] = (
0.2f * s_data[s_index1] +               //itself s_ind_1
0.1f * s_data[s_index0 -1] +       //N s_ind_0
0.1f * s_data[s_index0 +1] +       //NE s_ind_0
0.1f * s_data[s_index0   ] +       //E s_ind1
0.1f * s_data[s_index1 +1] +       //SE s_ind2
0.1f * s_data[s_index1 -1] +       //S s_ind2
0.1f * s_data[s_index2   ] +       //SW
0.1f * s_data[s_index2 -1] +       //W
0.1f * s_data[s_index2 +1]         //NW
) * 0.95f;//*/
}