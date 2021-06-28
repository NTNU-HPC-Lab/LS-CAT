#include "includes.h"
__global__ void drift(float *d_i1, float *d_icorr, int m1, int n1, int o1)
{
//grab theadID location
int tx = threadIdx.x;
int ty = threadIdx.y;

// get output values based on block and thread locations
int row_out = blockIdx.y*blockDim.y + ty;
int col_out = blockIdx.x*blockDim.x + tx;
int im_out  = blockIdx.z;

// Get starting value for the convolution as dictated by m2 and n2
// we'll use i1 indicies as the coord syst.
int row_st = row_out - (m1 - 1);
int col_st = col_out - (n1 - 1);

// correlation variable
float corr=0; // initialize correlation variable

if (row_out >= 0 && row_out < 2*m1 - 1 && col_out >= 0 && col_out < 2*n1- 1 && im_out < o1-1)  // ensure output is within bounds of correlation image
{
// Buffering into memory would be 1 call to a global variable, From there we need 1 call for each multiplication, however we only need to make 1 call to a global
// variable for the multiplication and move on, as such it doesn't make sense to buffer these images into local memory
for (int i = 0; i < m1; i++) { //
for (int j = 0; j < n1; j++)
{
if (row_st + i >= 0 && row_st + i < m1 && col_st + j >= 0 && col_st + j < n1) { // if row start and col start are greater than 0 and less than the number of pixels available perform convolution
corr += d_i1[row_st + i + (col_st + j) * m1 + im_out*m1*n1] * d_i1[i + j * m1 + (im_out+1)*m1*n1]; // shift n+1 image over n image
}
else {} // if else is invoked it's because row_st and col_st are outside of im1 bounds and the convolution should be left alone
}
}
d_icorr[row_out + col_out*(2*m1 - 1) + im_out*(2*m1 - 1)*(2*n1 - 1)] = corr; // assign correlation variable to proper location in final image
}
else{}


}