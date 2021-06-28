#include "includes.h"
__global__ void MedianFilterWithMask3x3_Kernel(float* output, const float* input, const int width, const int height, const int nChannels, const bool* keep_mask)
{
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

if (x >= width || y >= height)
return;
int offset = y*width + x;
if (keep_mask[offset])
{
for (int c = 0; c < nChannels; c++)
output[offset*nChannels + c] = input[offset*nChannels + c];
return;
}

float vals[9] = { 0 };
int count = 0;
for (int c = 0; c < nChannels; c++)
{
count = 0;
int start_x = ((x - 1) >= 0) ? (x - 1) : 0;
int end_x = ((x + 1) <= (width - 1)) ? (x + 1) : (width - 1);
int start_y = ((y - 1) >= 0) ? (y - 1) : 0;
int end_y = ((y + 1) <= (height - 1)) ? (y + 1) : (height - 1);
for (int ii = start_y; ii <= end_y; ii++)
{
for (int jj = start_x; jj <= end_x; jj++)
{
int cur_offset = ii*width + jj;
if (keep_mask[cur_offset])
{
vals[count++] = input[cur_offset*nChannels + c];
}
}
}
if (count == 0)
{
output[offset*nChannels + c] = 0;
}
else
{
int mid = (count + 1) / 2;
for (int pass = 0; pass < mid; pass++)
{
float max_val = vals[pass];
int max_id = pass;
for (int id = pass + 1; id < count; id++)
{
if (max_val < vals[id])
{
max_val = vals[id];
max_id = id;
}
}
vals[max_id] = vals[pass];
vals[pass] = max_val;
}
output[offset*nChannels + c] = vals[mid];
}
}
}