#include "includes.h"
__device__ inline float devIoU(float const *const a, float const *const b)
{
float left = max(a[0], b[0]), right = min(a[2], b[2]);
float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
float interS = width * height;
float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
return interS / (Sa + Sb - interS);
}
__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh, const float *dev_boxes, unsigned long long *dev_mask, const int NUM_BOX_CORNERS)
{
const int row_start = blockIdx.y;
const int col_start = blockIdx.x;

const int block_threads = blockDim.x;

const int row_size =
min(n_boxes - row_start * block_threads, block_threads);
const int col_size =
min(n_boxes - col_start * block_threads, block_threads);

__shared__ float block_boxes[NUM_THREADS_MACRO * NUM_2D_BOX_CORNERS_MACRO];
if (threadIdx.x < col_size)
{
block_boxes[threadIdx.x * NUM_BOX_CORNERS + 0] = dev_boxes[(block_threads * col_start + threadIdx.x) * NUM_BOX_CORNERS + 0];
block_boxes[threadIdx.x * NUM_BOX_CORNERS + 1] = dev_boxes[(block_threads * col_start + threadIdx.x) * NUM_BOX_CORNERS + 1];
block_boxes[threadIdx.x * NUM_BOX_CORNERS + 2] = dev_boxes[(block_threads * col_start + threadIdx.x) * NUM_BOX_CORNERS + 2];
block_boxes[threadIdx.x * NUM_BOX_CORNERS + 3] = dev_boxes[(block_threads * col_start + threadIdx.x) * NUM_BOX_CORNERS + 3];
}
__syncthreads();

if (threadIdx.x < row_size)
{
const int cur_box_idx = block_threads * row_start + threadIdx.x;
const float cur_box[NUM_2D_BOX_CORNERS_MACRO] = {dev_boxes[cur_box_idx*NUM_BOX_CORNERS + 0],
dev_boxes[cur_box_idx*NUM_BOX_CORNERS + 1],
dev_boxes[cur_box_idx*NUM_BOX_CORNERS + 2],
dev_boxes[cur_box_idx*NUM_BOX_CORNERS + 3]};
unsigned long long t = 0;
int start = 0;
if (row_start == col_start)
{
start = threadIdx.x + 1;
}
for (int i = start; i < col_size; i++)
{
if (devIoU(cur_box, block_boxes + i * NUM_BOX_CORNERS) > nms_overlap_thresh)
{
t |= 1ULL << i;
}
}
const int col_blocks = DIVUP(n_boxes, block_threads);
dev_mask[cur_box_idx * col_blocks + col_start] = t;
}
}