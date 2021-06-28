#include "includes.h"
__global__ void gpu_stencil37_hack2_cp_cols(double * dst, double * shared_rows, double *shared_cols,double *shared_slices,int d_xpitch, int d_ypitch, int d_zpitch,int s_xpitch,int s_ypitch, int s_zpitch, int n_rows, int n_cols,int n_slices,int tile_x,int tile_y, int tile_z){

#ifdef CUDA_CUDA_DEBUG
if((blockIdx.x==0)&&(blockIdx.y==0)&&(blockIdx.z==0)&&(threadIdx.y==0)&& threadIdx.x==0 && threadIdx.z==0){
printf("copy cols: begin\n");
printf("copy cols: n_cols=%d,n_rows=%d,n_slices=%d\n",n_cols,n_rows,n_slices);
printf("copy cols: gridDim.x=%d,gridDim.y=%d,gridDim.z=%d\n",gridDim.x,gridDim.y,gridDim.z);
printf("copy cols: blockDim.x=%d,blockDim.y=%d,blockDim.z=%d\n",blockDim.x,blockDim.y,blockDim.z);
printf("copy cols: tile_x=%d,tile_y=%d,tile_z=%d\n",tile_x,tile_y,tile_z);
}
#endif
int base_global_slice = tile_z * blockIdx.z;
int base_global_row   = blockDim.y * blockIdx.y;
int base_global_col   = tile_x * blockIdx.x;

//int dst_area = n_rows*n_cols;
//int shared_area = gridDim.x*n_rows*2;
int dst_area = d_ypitch*d_xpitch;
int shared_area = gridDim.x*s_ypitch*2;

#ifdef CUDA_CUDA_DEBUG
if((blockIdx.x==0)&&(blockIdx.y==0)&&(blockIdx.z==0)&&(threadIdx.y==0&&threadIdx.x==0&&threadIdx.z==0)){
printf("copy cols: shared_area=%d\n",shared_area);
}
#endif
//int base_global_idx = base_global_slice*dst_area + base_global_row * n_cols + base_global_col;
int base_global_idx = base_global_slice*dst_area + base_global_row * d_xpitch + base_global_col;

int nextCol= base_global_col+1;
bool legalNextCol = (nextCol<n_cols)?1:0;

int ty = threadIdx.y;
bool legalCurRow = (base_global_row + ty)<n_rows;

for(int tz=0;tz<tile_z;++tz){
bool legalCurSlice = (base_global_slice + tz)<n_slices;
//int idx = (base_global_slice+tz)*shared_area + blockIdx.x*2*n_rows+blockIdx.y*blockDim.y+ty;
//int idx_dst =base_global_idx + tz*dst_area + ty*n_cols ;
int idx = (base_global_slice+tz)*shared_area + blockIdx.x*2*s_ypitch+blockIdx.y*blockDim.y+ty;
int idx_dst =base_global_idx + tz*dst_area + ty*d_xpitch ;

if(legalCurRow && legalCurSlice){
shared_cols[idx] = dst[idx_dst];
}
if(legalCurRow && legalCurSlice && legalNextCol){
//shared_cols[idx + n_rows] = dst[idx_dst + 1];
shared_cols[idx + s_ypitch] = dst[idx_dst + 1];
}

__syncthreads();
}
__syncthreads();

#ifdef CUDA_CUDA_DEBUG
if(blockIdx.z ==0 && blockIdx.y==0 && blockIdx.x==0 && (threadIdx.x==0)){
//		printf("shared_cols: addr:%d, val = %f\n", threadIdx.y,shared_cols[threadIdx.y]);
}

if(blockIdx.y==0 && blockIdx.x==0 &&blockIdx.z==0 ){
if((threadIdx.x==0 || threadIdx.x==1 || threadIdx.x==2 ) && threadIdx.y==0){

int d_addr0 = base_global_idx+0*dst_area+threadIdx.x;
int d_addr1 = base_global_idx+1*dst_area+threadIdx.x;
int addr  = base_global_slice+blockIdx.x*blockDim.x + threadIdx.x;
int addr1 = shared_area*(base_global_slice+1)+blockIdx.x*blockDim.x+ threadIdx.x;
int addr2 = shared_area*(base_global_slice+2)+blockIdx.x*blockDim.x+ threadIdx.x;
printf("copy cols: blockIdx.x=%d, blockIdx.y=%d,blockIdx.z=%d,dst        : z:%d, addr:%d, val = %f\n",blockIdx.x, blockIdx.y,blockIdx.z,0,d_addr0,dst[d_addr0]);
printf("copy cols: blockIdx.x=%d, blockIdx.y=%d,blockIdx.z=%d,dst        : z:%d, addr:%d, val = %f\n",blockIdx.x, blockIdx.y,blockIdx.z,1,d_addr1,dst[d_addr1]);
printf("copy cols: blockIdx.x=%d, blockIdx.y=%d,blockIdx.z=%d,shared_cols: z:%d, addr:%d, val = %f\n",blockIdx.x, blockIdx.y,blockIdx.z,0,addr,shared_cols[addr]);
printf("copy cols: blockIdx.x=%d, blockIdx.y=%d,blockIdx.z=%d,shared_cols: z:%d, addr:%d, val = %f\n",blockIdx.x, blockIdx.y,blockIdx.z,1,addr1,shared_cols[addr1]);
printf("copy cols: blockIdx.x=%d, blockIdx.y=%d,blockIdx.z=%d,shared_cols: z:%d, addr:%d, val = %f\n",blockIdx.x, blockIdx.y,blockIdx.z,2,addr2,shared_cols[addr2]);
}
}

#endif

#ifdef CUDA_CUDA_DEBUG
if((blockIdx.x==0)&&(blockIdx.y==0)&&(blockIdx.z==0)&&(threadIdx.y==0 && threadIdx.x==0 && threadIdx.z==0)){
printf("copy cols end!\n");
}
#endif
}