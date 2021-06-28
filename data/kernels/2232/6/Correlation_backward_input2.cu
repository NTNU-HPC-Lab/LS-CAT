#include "includes.h"
__global__ void Correlation_backward_input2(int item, float *gradInput2, int nInputChannels, int inputHeight, int inputWidth, float *gradOutput, int nOutputChannels, int outputHeight, int outputWidth, float *rInput1, int pad_size, int kernel_size, int max_displacement, int stride1, int stride2)
{
// n (batch size), c (num of channels), y (height), x (width)

int n = item;
int y = blockIdx.x * stride1 + pad_size;
int x = blockIdx.y * stride1 + pad_size;
int c = blockIdx.z;

int tch_off = threadIdx.x;

int kernel_rad = (kernel_size - 1) / 2;
int displacement_rad = max_displacement / stride2;
int displacement_size = 2 * displacement_rad + 1;

int pInputWidth = inputWidth + 2 * pad_size;
int pInputHeight = inputHeight + 2 * pad_size;

int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
int pdimxc = pInputWidth * nInputChannels;
int pdimc = nInputChannels;

int tdimcyx = nOutputChannels * outputHeight * outputWidth;
int tdimyx = outputHeight * outputWidth;
int tdimx = outputWidth;

int odimcyx = nInputChannels * inputHeight* inputWidth;
int odimyx = inputHeight * inputWidth;
int odimx = inputWidth;

float nelems = kernel_size * kernel_size * nInputChannels;

__shared__ float prod_sum[CUDA_NUM_THREADS];
prod_sum[tch_off] = 0;

for (int tc = tch_off; tc < nOutputChannels; tc += CUDA_NUM_THREADS) {
int i2 = (tc % displacement_size - displacement_rad) * stride2;
int j2 = (tc / displacement_size - displacement_rad) * stride2;

int xmin = (x - kernel_rad - max_displacement - i2) / stride1;
int ymin = (y - kernel_rad - max_displacement - j2) / stride1;

int xmax = (x + kernel_rad - max_displacement - i2) / stride1;
int ymax = (y + kernel_rad - max_displacement - j2) / stride1;

if (xmax < 0 || ymax < 0 || xmin >= outputWidth || ymin >= outputHeight) {
// assumes gradInput2 is pre-allocated and zero filled
continue;
}

if (xmin > xmax || ymin > ymax) {
// assumes gradInput2 is pre-allocated and zero filled
continue;
}

xmin = max(0,xmin);
xmax = min(outputWidth-1,xmax);

ymin = max(0,ymin);
ymax = min(outputHeight-1,ymax);

int indx1 = n * pdimyxc + (y - j2)* pdimxc + (x - i2) * pdimc + c;
float val1 = rInput1[indx1];

for (int j = ymin; j <= ymax; ++j) {
for (int i = xmin; i <= xmax; ++i) {
int tindx = n * tdimcyx + tc * tdimyx + j * tdimx + i;
prod_sum[tch_off] += gradOutput[tindx] * val1;
}
}
}

__syncthreads();

if(tch_off == 0) {
float reduce_sum = 0;
for(int idx = 0; idx < CUDA_NUM_THREADS; idx++) {
reduce_sum += prod_sum[idx];
}
const int indx2 = n * odimcyx + c * odimyx + (y - pad_size) * odimx + (x - pad_size);
gradInput2[indx2] = reduce_sum / nelems;
}

}