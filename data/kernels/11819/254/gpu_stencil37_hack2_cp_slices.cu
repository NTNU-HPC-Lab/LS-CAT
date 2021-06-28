#include "includes.h"
__global__ void gpu_stencil37_hack2_cp_slices(double * dst, double * shared_rows, double *shared_cols,double *shared_slices,int d_xpitch,int d_ypitch,int d_zpitch,int s_xpitch,int s_ypitch, int s_zpitch, int n_rows, int n_cols,int n_slices, int tile_x,int tile_y, int tile_z){

#ifdef CUDA_DARTS_DEBUG
if((blockIdx.x==0)&&(blockIdx.y==0)&&(blockIdx.z==0)&&(threadIdx.x==0)){
printf("copy slices: begin!\n");
printf("copy slices: n_cols=%d,n_rows=%d,n_slices=%d\n",n_cols,n_rows,n_slices);
printf("copy slices: gridDim.x=%d,gridDim.y=%d,gridDim.z=%d\n",gridDim.x,gridDim.y,gridDim.z);
printf("copy slices: blockDim.x=%d,blockDim.y=%d,blockDim.z=%d\n",blockDim.x,blockDim.y,blockDim.z);
printf("copy slices: tile_x=%d,tile_y=%d,tile_z=%d\n",tile_x,tile_y,tile_z);
}
#endif
int base_global_slice = tile_z * blockIdx.z;
int base_global_row   = tile_y * blockIdx.y;
int base_global_col   = blockDim.x * blockIdx.x;

//int area = n_rows*n_cols;
//int base_global_idx = base_global_slice*area + base_global_row * n_cols + base_global_col;
//int d_area = n_rows*d_xpitch;
//int s_area = n_rows*n_cols;
int d_area = d_ypitch*d_xpitch;
int s_area = s_ypitch*s_xpitch;
int base_global_idx = base_global_slice*d_area + base_global_row * d_xpitch + base_global_col;

int nextSlice = base_global_slice+1;
bool legalNextSlice = (nextSlice<n_slices);
int tx = threadIdx.x;
bool legalCurCol = (base_global_col + tx)<n_cols;

for(int ty=0;ty<tile_y;++ty){
bool legalCurRow = (base_global_row + ty)<n_rows;
//int s_idx = blockIdx.z*s_area*2 + (base_global_row+ty)*n_cols + base_global_col+tx ;
//int dst_idx = base_global_idx + ty*n_cols+tx;
int s_idx = blockIdx.z*s_area*2 + (base_global_row+ty)*s_xpitch + base_global_col+tx ;
int d_idx = base_global_idx + ty*d_xpitch+tx;
if(legalCurCol&&legalCurRow){
shared_slices[s_idx] = dst[d_idx];
}
if(legalNextSlice&&legalCurCol&&legalCurRow){
shared_slices[s_idx+s_area] = dst[d_idx+d_area];
}

}
__syncthreads();

#ifdef CUDA_CUDA_DEBUG
if(blockIdx.z ==0 && blockIdx.y==0 && blockIdx.x==0 ){
//	printf("shared_slices: addr:%d, val = %f\n",n_cols*n_rows + threadIdx.x,shared_slices[n_cols*n_rows+threadIdx.x]);
if(threadIdx.x==0||threadIdx.x==1||threadIdx.x==2){
int addr  = s_xpitch*s_ypitch + blockDim.x*blockIdx.x+threadIdx.x;
int addr1 = s_xpitch*s_ypitch + blockDim.x*blockIdx.x+threadIdx.x+s_xpitch;
int addr2 = s_xpitch*s_ypitch + blockDim.x*blockIdx.x+threadIdx.x+s_xpitch*2;

int daddr  = d_xpitch*d_ypitch + blockDim.x*blockIdx.x+threadIdx.x;
int daddr1 = d_xpitch*d_ypitch + blockDim.x*blockIdx.x+threadIdx.x+d_xpitch;
int daddr2 = d_xpitch*d_ypitch + blockDim.x*blockIdx.x+threadIdx.x+d_xpitch*2;
printf("copy slices: blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d,dst: addr= %d, val= %f\n",blockIdx.x, blockIdx.y, blockIdx.z, daddr,dst[daddr]);
printf("copy slices: blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d,dst: addr= %d, val= %f\n",blockIdx.x, blockIdx.y, blockIdx.z, daddr1,dst[daddr1]);
printf("copy slices: blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d,dst: addr= %d, val= %f\n",blockIdx.x, blockIdx.y, blockIdx.z, daddr2,dst[daddr2]);

printf("copy slices: blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d,shared_slices: addr= %d, val= %f\n",blockIdx.x, blockIdx.y, blockIdx.z, addr,shared_slices[addr]);
printf("copy slices: blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d,shared_slices: addr= %d, val= %f\n",blockIdx.x, blockIdx.y, blockIdx.z, addr1,shared_slices[addr1]);
printf("copy slices: blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d,shared_slices: addr= %d, val= %f\n",blockIdx.x, blockIdx.y, blockIdx.z, addr2,shared_slices[addr2]);
}
}
#endif

#ifdef CUDA_DARTS_DEBUG
if((blockIdx.x==0)&&(blockIdx.y==0)&&(blockIdx.z==0)&&(threadIdx.x==0)){
printf("copy slices end!\n");
}
#endif
}