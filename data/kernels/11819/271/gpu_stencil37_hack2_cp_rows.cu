#include "includes.h"
__global__ void gpu_stencil37_hack2_cp_rows(double * dst, double * shared_rows, double *shared_cols,double *shared_slices,int d_xpitch,int d_ypitch,int d_zpitch,int s_xpitch,int s_ypitch, int s_zpitch, int n_rows, int n_cols,int n_slices,int tile_x,int tile_y, int tile_z){

#ifdef CUDA_DARTS_DEBUG
if((blockIdx.x==0)&&(blockIdx.y==0)&&(blockIdx.z==0)&&(threadIdx.x==0)){
printf("copy rows: begin\n");
printf("copy rows: n_cols=%d,n_rows=%d,n_slices=%d\n",n_cols,n_rows,n_slices);
printf("copy rows: gridDim.x=%d,gridDim.y=%d,gridDim.z=%d\n",gridDim.x,gridDim.y,gridDim.z);
printf("copy rows: blockDim.x=%d,blockDim.y=%d,blockDim.z=%d\n",blockDim.x,blockDim.y,blockDim.z);
printf("copy rows: tile_x=%d,tile_y=%d,tile_z=%d\n",tile_x,tile_y,tile_z);
}
#endif
int base_global_slice = tile_z * blockIdx.z;
int base_global_row   = tile_y  * blockIdx.y;
int base_global_col   = blockDim.x*blockIdx.x;

//int dst_area = n_rows*n_cols;
//int s_area = gridDim.y*n_cols*2;
int dst_area = d_ypitch*d_xpitch;
int s_area = gridDim.y*s_xpitch*2;

//int base_global_idx = base_global_slice*dst_area + base_global_row * n_cols + base_global_col;
int base_global_idx = base_global_slice*dst_area + base_global_row * d_xpitch + base_global_col;

int nextRow = base_global_row+1;
bool legalNextRow = nextRow<n_rows;

int tx = threadIdx.x;
bool legalCurCol = (base_global_col + tx)<n_cols;

for(int tz=0;tz<tile_z;++tz){
bool legalCurSlice = (base_global_slice + tz)<n_slices;
int idx_dst =base_global_idx + tz*dst_area+ tx  ;
//int idx = (base_global_slice+tz)*s_area + blockIdx.y*n_cols*2+blockIdx.x*blockDim.x+ tx  ;
int idx = (base_global_slice+tz)*s_area + blockIdx.y*s_xpitch*2+blockIdx.x*blockDim.x+ tx  ;
if(legalCurCol && legalCurSlice){
shared_rows[idx] = dst[idx_dst];
}
if(legalCurCol && legalCurSlice && legalNextRow){
//shared_rows[idx+n_cols] = dst[idx_dst+n_cols];
shared_rows[idx+s_xpitch] = dst[idx_dst+d_xpitch];
}


}
__syncthreads();

#ifdef CUDA_CUDA_DEBUG
if(blockIdx.y==0 && blockIdx.x==0 &&blockIdx.z==0 ){
if((threadIdx.x==0 || threadIdx.x==1 || threadIdx.x==2 ) && threadIdx.y==0){

int d_addr0 = base_global_idx+0*dst_area+threadIdx.x;
int d_addr1 = base_global_idx+1*dst_area+threadIdx.x;
int s_addr00  = base_global_slice+blockIdx.x*blockDim.x + threadIdx.x;
int s_addr01  = base_global_slice+blockIdx.x*blockDim.x + threadIdx.x+s_xpitch;
int s_addr02  = base_global_slice+blockIdx.x*blockDim.x + threadIdx.x+s_xpitch*2;
int s_addr10 = s_area*(base_global_slice+1)+blockIdx.x*blockDim.x+ threadIdx.x;
int s_addr11 = s_area*(base_global_slice+1)+blockIdx.x*blockDim.x+ threadIdx.x+s_xpitch;
int s_addr12 = s_area*(base_global_slice+1)+blockIdx.x*blockDim.x+ threadIdx.x+s_xpitch*2;
int s_addr20 = s_area*(base_global_slice+2)+blockIdx.x*blockDim.x+ threadIdx.x;
int s_addr21 = s_area*(base_global_slice+2)+blockIdx.x*blockDim.x+ threadIdx.x+s_xpitch;
int s_addr22 = s_area*(base_global_slice+2)+blockIdx.x*blockDim.x+ threadIdx.x+s_xpitch*2;
printf("copy rows: blockIdx.x=%d, blockIdx.y=%d,blockIdx.z=%d,dst        : z:%d, addr:%d, val = %f\n",blockIdx.x, blockIdx.y,blockIdx.z,0,d_addr0,dst[d_addr0]);
printf("copy rows: blockIdx.x=%d, blockIdx.y=%d,blockIdx.z=%d,dst        : z:%d, addr:%d, val = %f\n",blockIdx.x, blockIdx.y,blockIdx.z,1,d_addr1,dst[d_addr1]);
printf("copy rows: blockIdx.x=%d, blockIdx.y=%d,blockIdx.z=%d,shared_rows: z:%d, addr:%d, val = %f\n",blockIdx.x, blockIdx.y,blockIdx.z,0,s_addr00,shared_rows[s_addr00]);
printf("copy rows: blockIdx.x=%d, blockIdx.y=%d,blockIdx.z=%d,shared_rows: z:%d, addr:%d, val = %f\n",blockIdx.x, blockIdx.y,blockIdx.z,1,s_addr01,shared_rows[s_addr01]);
printf("copy rows: blockIdx.x=%d, blockIdx.y=%d,blockIdx.z=%d,shared_rows: z:%d, addr:%d, val = %f\n",blockIdx.x, blockIdx.y,blockIdx.z,2,s_addr00,shared_rows[s_addr02]);
printf("copy rows: blockIdx.x=%d, blockIdx.y=%d,blockIdx.z=%d,shared_rows: z:%d, addr:%d, val = %f\n",blockIdx.x, blockIdx.y,blockIdx.z,1,s_addr10,shared_rows[s_addr10]);
printf("copy rows: blockIdx.x=%d, blockIdx.y=%d,blockIdx.z=%d,shared_rows: z:%d, addr:%d, val = %f\n",blockIdx.x, blockIdx.y,blockIdx.z,1,s_addr11,shared_rows[s_addr11]);
printf("copy rows: blockIdx.x=%d, blockIdx.y=%d,blockIdx.z=%d,shared_rows: z:%d, addr:%d, val = %f\n",blockIdx.x, blockIdx.y,blockIdx.z,1,s_addr12,shared_rows[s_addr12]);
printf("copy rows: blockIdx.x=%d, blockIdx.y=%d,blockIdx.z=%d,shared_rows: z:%d, addr:%d, val = %f\n",blockIdx.x, blockIdx.y,blockIdx.z,2,s_addr20,shared_rows[s_addr20]);
printf("copy rows: blockIdx.x=%d, blockIdx.y=%d,blockIdx.z=%d,shared_rows: z:%d, addr:%d, val = %f\n",blockIdx.x, blockIdx.y,blockIdx.z,2,s_addr21,shared_rows[s_addr21]);
printf("copy rows: blockIdx.x=%d, blockIdx.y=%d,blockIdx.z=%d,shared_rows: z:%d, addr:%d, val = %f\n",blockIdx.x, blockIdx.y,blockIdx.z,2,s_addr22,shared_rows[s_addr22]);
}
if(threadIdx.x==0 && threadIdx.y==0){
int addr =  2*s_area+n_cols+256;
int addr1 = 2*dst_area+n_cols+256;
printf("copy rows: shared_rows: addr:%d, val:%f\n", addr, shared_rows[addr]);
printf("copy rows: dst        : addr:%d, val:%f\n", addr1, dst[addr1]);
}
}
#endif

#ifdef CUDA_DARTS_DEBUG

if((blockIdx.x==0)&&(blockIdx.y==0)&&(blockIdx.z==0)&&(threadIdx.x==0)){
printf("copy rows end!\n");
}
#endif
}