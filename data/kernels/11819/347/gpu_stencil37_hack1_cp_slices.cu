#include "includes.h"
__global__ void gpu_stencil37_hack1_cp_slices(double * dst, double * shared_rows, double *shared_cols,double *shared_slices,uint64_t n_rows, uint64_t n_cols,uint64_t n_slices,int tile_x,int tile_y, int tile_z){

#ifdef CUDA_DARTS_DEBUG
if((blockIdx.x==0)&&(blockIdx.y==0)&&(blockIdx.z==0)&&(threadIdx.x==0)){
printf("copy slices begin!\n");
printf("gridDim.x=%d,gridDim.y=%d,gridDim.z=%d\n",gridDim.x,gridDim.y,gridDim.z);
printf("blockDim.x=%d,blockDim.y=%d,blockDim.z=%d\n",blockDim.x,blockDim.y,blockDim.z);
printf("tile_x=%d,tile_y=%d,tile_z=%d\n",tile_x,tile_y,tile_z);
}
#endif
int base_global_slice = tile_z * blockIdx.z;
int base_global_row   = tile_y * blockIdx.y;
int base_global_col   = blockDim.x * blockIdx.x;

uint64_t area = n_rows*n_cols;
uint64_t base_global_idx = base_global_slice*area + base_global_row * n_cols + base_global_col;

int nextSlice = base_global_slice+1;
bool legalNextSlice = (nextSlice<n_slices);
int tx = threadIdx.x;
bool legalCurCol = (base_global_col + tx)<n_cols;

for(int ty=0;ty<tile_y;++ty){
bool legalCurRow = (base_global_row + ty)<n_rows;
uint64_t idx = blockIdx.z*area*2 + (base_global_row+ty)*n_cols + base_global_col+tx ;
uint64_t idx_dst = base_global_idx + ty*n_cols+tx;
if(legalCurCol&&legalCurRow){
shared_slices[idx] = dst[idx_dst];
}
if(legalNextSlice&&legalCurCol&&legalCurRow){
shared_slices[idx+area] = dst[idx_dst+area];
}

}
__syncthreads();

#ifdef CUDA_CUDA_DEBUG
if(blockIdx.z ==0 && blockIdx.y==0 && blockIdx.x==1 ){
//	printf("shared_slices: addr:%d, val = %f\n",n_cols*n_rows + threadIdx.x,shared_slices[n_cols*n_rows+threadIdx.x]);
if(threadIdx.x==0||threadIdx.x==1||threadIdx.x==2){
int addr = n_cols*n_rows + blockDim.x*blockIdx.x+threadIdx.x;
int addr1 = n_cols*n_rows + blockDim.x*blockIdx.x+threadIdx.x+n_cols;
int addr2 = n_cols*n_rows + blockDim.x*blockIdx.x+threadIdx.x+n_cols*2;
printf("blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d,shared_slices: addr= %d, val= %f\n",blockIdx.x, blockIdx.y, blockIdx.z, addr,shared_slices[addr]);
printf("blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d,shared_slices: addr= %d, val= %f\n",blockIdx.x, blockIdx.y, blockIdx.z, addr1,shared_slices[addr1]);
printf("blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d,shared_slices: addr= %d, val= %f\n",blockIdx.x, blockIdx.y, blockIdx.z, addr2,shared_slices[addr2]);
}
}
#endif

#ifdef CUDA_DARTS_DEBUG
if((blockIdx.x==0)&&(blockIdx.y==0)&&(blockIdx.z==0)&&(threadIdx.x==0)){
printf("copy slices end!\n");
}
#endif
}