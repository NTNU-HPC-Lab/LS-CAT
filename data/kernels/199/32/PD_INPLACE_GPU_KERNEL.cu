#include "includes.h"
__global__ void PD_INPLACE_GPU_KERNEL(float *d_input, float *d_temp, unsigned char *d_output_taps, float *d_MSD, int maxTaps, int nTimesamples)
{
extern __shared__ float s_input[]; //dynamically allocated memory for now

int f, i, gpos_y, gpos_x, spos, itemp;
float res_SNR[PD_NWINDOWS], SNR, temp_FIR_value, FIR_value, ftemp;
int res_Taps[PD_NWINDOWS];
float signal_mean, signal_sd, modifier;
signal_mean = d_MSD[0];
signal_sd = d_MSD[2];
modifier = d_MSD[1];

//----------------------------------------------
//----> Reading data
gpos_y = blockIdx.y * nTimesamples;
gpos_x = blockIdx.x * PD_NTHREADS * PD_NWINDOWS + threadIdx.x;
spos = threadIdx.x;
for (f = 0; f < PD_NWINDOWS; f++)
{
if (gpos_x < nTimesamples)
{
s_input[spos] = d_input[gpos_y + gpos_x];
}
spos = spos + blockDim.x;
gpos_x = gpos_x + blockDim.x;
}

//----> Loading shared data
itemp = PD_NTHREADS * PD_NWINDOWS + maxTaps - 1;
gpos_y = blockIdx.y * ( maxTaps - 1 ) * gridDim.x;
gpos_x = blockIdx.x * ( maxTaps - 1 ) + threadIdx.x;
while (spos < itemp)
{ // && gpos_x<((maxTaps-1)*gridDim.x)
s_input[spos] = d_temp[gpos_y + gpos_x];
spos = spos + blockDim.x;
gpos_x = gpos_x + blockDim.x;
}

__syncthreads();

//----> SNR for nTaps=1
spos = PD_NWINDOWS * threadIdx.x;
for (i = 0; i < PD_NWINDOWS; i++)
{
res_SNR[i] = ( s_input[spos + i] - signal_mean ) / signal_sd;
res_Taps[i] = 1;
}

//----------------------------------------------
//----> FIR calculation loop
FIR_value = s_input[spos];
for (f = 1; f < maxTaps; f++)
{
//nTaps=f+1;!
ftemp = signal_sd + f * modifier;
spos = PD_NWINDOWS * threadIdx.x;

// 0th element from NWINDOW
i = 0;
FIR_value += s_input[spos + f];

SNR = ( FIR_value - ( f + 1 ) * signal_mean ) / ( ftemp );
if (SNR > res_SNR[i])
{
res_SNR[i] = SNR;
res_Taps[i] = f + 1;
}

temp_FIR_value = FIR_value;
for (i = 1; i < PD_NWINDOWS; i++)
{
temp_FIR_value = temp_FIR_value - s_input[spos + i - 1] + s_input[spos + f + i];

SNR = ( temp_FIR_value - ( f + 1 ) * signal_mean ) / ( ftemp );
if (SNR > res_SNR[i])
{
res_SNR[i] = SNR;
res_Taps[i] = f + 1;
}
}
}

//----------------------------------------------
//---- Writing data
gpos_y = blockIdx.y * nTimesamples;
gpos_x = blockIdx.x * PD_NTHREADS * PD_NWINDOWS + PD_NWINDOWS * threadIdx.x;
for (i = 0; i < PD_NWINDOWS; i++)
{
if (( gpos_x + i ) < ( nTimesamples ))
{
d_input[gpos_y + gpos_x + i] = res_SNR[i];
d_output_taps[gpos_y + gpos_x + i] = res_Taps[i];
}
}
}