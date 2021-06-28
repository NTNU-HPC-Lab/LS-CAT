#include "includes.h"
__device__ inline float devIoU(const float *a, const float *b){
//a: [5, ] b: [5, ], ymin, xmin, ymax, xmax, score
float w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]));
float h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]));
float intersect = w * h;
float sa = (a[2] - a[0]) * (a[3] - a[1]);
float sb = (b[2] - b[0]) * (b[3] - b[1]);
float _union = sa + sb - intersect;
float eps = 1e-4;
return intersect * 1.0 / (_union + eps);
}
__global__ void nms_kernel(float *bbox_dev, unsigned long long *mask_dev, int num_boxes, int col_blocks, float threshold){
//for each block(c, r) with thread(t, 0), compute the cur_box: r * 64 + t with boxes[c*64 to c*64+63], store to mask_dev
//bx = c, by = r, t = tx
int bx = blockIdx.x;
int by = blockIdx.y;
int tx = threadIdx.x;

//因为划block时取整，最后一组可能不满, 实际上的row对应block上面的y方向
const int row_size = min(num_boxes - by * THREADS, THREADS);
const int col_size = min(num_boxes - bx * THREADS, THREADS);

__shared__ float sh[THREADS * 5];
//put [c*64 ~ c*64+63] to share mem, i.e., in parallel: c * 64 + bx, 放入的时候可以并行放
if(tx < col_size){
int cols = tx + bx * THREADS;
#pragma unroll 5
for(int j = 0; j < 5; j++){
sh[tx * 5 + j] = bbox_dev[cols * 5 + j];
}
__syncthreads();
}

//compute cur_box at each row: r * 64 + t with shared mem
if(tx < row_size){
//compute cur with share mem
const int cur_box_idx = (by * THREADS) + tx;
float *cur_box = bbox_dev + cur_box_idx * 5;

int start = 0;
if(bx == by){
start = tx + 1;
}

unsigned long long t = 0;
for(int i = start; i < col_size; i++){
if(devIoU(cur_box, sh + tx * 5) >= threshold){
t |= (1ULL<<tx);
}
}

const int mask_idx = cur_box_idx * col_blocks + bx;
mask_dev[mask_idx] = t;
}
}