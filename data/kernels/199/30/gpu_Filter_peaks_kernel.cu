#include "includes.h"
__global__ void gpu_Filter_peaks_kernel(unsigned int *d_new_peak_list_DM, unsigned int *d_new_peak_list_TS, unsigned int *d_new_peak_list_BW, float *d_new_peak_list_SNR, unsigned int *d_peak_list_DM, unsigned int *d_peak_list_TS, unsigned int *d_peak_list_BW, float *d_peak_list_SNR, unsigned int nElements, unsigned int max_distance, int nLoops, int max_list_pos, int *gmem_pos){
// PPF_DPB = 128 //this is because I set nThreads to 64
// PPF_PEAKS_PER_BLOCK = something small like 10
__shared__ float s_data_snr[PPF_DPB];
__shared__ int s_data_dm[PPF_DPB];
__shared__ int s_data_ts[PPF_DPB];
__shared__ int s_flag[PPF_NTHREADS];
int d, s;
int elements_pos, pos;
float snr, distance, fs, fd;
//	float4 f4temp;


if(threadIdx.x<PPF_PEAKS_PER_BLOCK){
s_flag[threadIdx.x] = 1;
}
else{
s_flag[threadIdx.x] = 0;
}


for(int f=0; f<nLoops; f++){
// Load new data blob
//s_data[threadIdx.x + 2*PPF_DPB] = 0; // SNR
//s_data[threadIdx.x + 64 + 2*PPF_DPB] = 0; // SNR

pos = PPF_DPB*f + threadIdx.x;
if(pos < nElements){
//			f4temp = __ldg(&d_peak_list[pos]);
s_data_dm[threadIdx.x]  = d_peak_list_DM[pos]; //f4temp.x; // DM
s_data_ts[threadIdx.x]  = d_peak_list_TS[pos]; //f4temp.y; // Time
s_data_snr[threadIdx.x] = d_peak_list_SNR[pos]; //f4temp.z; // SNR
}
else {
s_data_dm[threadIdx.x]  = 0; //f4temp.x; // DM
s_data_ts[threadIdx.x]  = 0; //f4temp.y; // Time
s_data_snr[threadIdx.x] = -1000; //f4temp.z; // SNR
}
//		if(blockIdx.x==0 && threadIdx.x==0) printf("point: [%d;%d;%lf]\n",  s_data_dm[threadIdx.x], s_data_ts[threadIdx.x], s_data_snr[threadIdx.x]);


pos = PPF_DPB*f + threadIdx.x + PPF_NTHREADS;
if(pos < nElements){
//			f4temp = __ldg(&d_peak_list[PPF_DPB*f + threadIdx.x + (PPF_DPB>>1)]);
s_data_dm[threadIdx.x + PPF_NTHREADS ] = d_peak_list_DM[pos]; //f4temp.x; // DM
s_data_ts[threadIdx.x + PPF_NTHREADS ] = d_peak_list_TS[pos]; //f4temp.y; // Time
s_data_snr[threadIdx.x + PPF_NTHREADS] = d_peak_list_SNR[pos]; //f4temp.z; // SNR
}
else {
s_data_dm[threadIdx.x + PPF_NTHREADS]  = 0; //f4temp.x; // DM
s_data_ts[threadIdx.x + PPF_NTHREADS]  = 0; //f4temp.y; // Time
s_data_snr[threadIdx.x + PPF_NTHREADS] = -1000; //f4temp.z; // SNR
}

__syncthreads();

elements_pos = blockIdx.x*PPF_PEAKS_PER_BLOCK;
for(int p=0; p<PPF_PEAKS_PER_BLOCK; p++){
//			if (blockIdx.x == 0) printf("%d %d\n", p, s_flag[p]);
if((s_flag[p]) && ((elements_pos + p) < nElements)){
//pos = elements_pos+p;
//if(pos<nElements){
d   = d_peak_list_DM[elements_pos+p]; // DM
s   = d_peak_list_TS[elements_pos+p]; // Time
snr = d_peak_list_SNR[elements_pos+p]; // SNR

// first element
//					if(blockIdx.x==0) printf("s_data: %lf, snr: %lf, pos: %d\n", s_data_snr[threadIdx.x], snr, p);
if( (s_data_snr[threadIdx.x] >= snr)){
fs = ((float)s_data_dm[threadIdx.x] - (float)d);
fd = ((float)s_data_ts[threadIdx.x] - (float)s);
distance = (fd*fd + fs*fs);
//						if(blockIdx.x==0) printf("%d - %d = %d; %d - %d = %d\n",s_data_dm[threadIdx.x], d, fs, s_data_ts[threadIdx.x], s, fd, distance);
if( (distance < (float)max_distance) && (distance!=0) ){
//							if(blockIdx.x==0) printf("distance: %d %lf %lf %lf %d %d;\n", p, distance, fs, fd, s, d);
s_flag[p]=0;
}
}

//second element
if(s_data_snr[threadIdx.x + PPF_NTHREADS] >= snr){
fs = ((float)s_data_dm[threadIdx.x + PPF_NTHREADS] - (float)d);
fd = ((float)s_data_ts[threadIdx.x + PPF_NTHREADS] - (float)s);
distance = (fd*fd + fs*fs);
//						if(blockIdx.x==0) printf("%d - %d = %d; %d - %d = %d\n",s_data_dm[threadIdx.x], d, fs, s_data_ts[threadIdx.x], s, fd, distance);
if( (distance < (float)max_distance) && (distance!=0)){
s_flag[p]=0;
//							if(blockIdx.x==0) printf("xdistance: %d %lf %lf %lf %d %d;\n", p, distance, fs, fd, s, d);
}
}
//}
}
} // for p

}

// Saving peaks that got through
elements_pos = blockIdx.x*PPF_PEAKS_PER_BLOCK;
if(threadIdx.x < PPF_PEAKS_PER_BLOCK){
if( (s_flag[threadIdx.x] == 1) && ((elements_pos + threadIdx.x) < nElements)){
int list_pos=atomicAdd(gmem_pos, 1);
if(list_pos<max_list_pos){
d_new_peak_list_DM[list_pos]  = d_peak_list_DM[elements_pos  + threadIdx.x];
d_new_peak_list_TS[list_pos]  = d_peak_list_TS[elements_pos  + threadIdx.x];
d_new_peak_list_BW[list_pos]  = d_peak_list_BW[elements_pos  + threadIdx.x];
d_new_peak_list_SNR[list_pos] = d_peak_list_SNR[elements_pos + threadIdx.x];
}
}
}
}