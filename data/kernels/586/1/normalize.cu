#include "includes.h"

/*
waveform.cu:°üº¬µÄº¯ÊýÖ÷ÒªÊÇ¶ÔÓ¦SpikeDetect²¿·ÖµÄwaveformµÄÒ»Ð©²Ù×÷
º¯Êý×÷ÓÃÈçÏÂ£º
comps_wave()£º¶ÔÓÚdetect²¿·ÖÌáÈ¡µ½µÄcomponents£¬´Ó±ä»»ºóµÄ²¨ÐÎdata_tÖÐÌáÈ¡¶ÔÓ¦µÄwave
normalize()£º¶ÔÓÚ²¨ÐÎÖÐµÄµçÎ»Öµ£¬Í¨¹ý¸ßãÐÖµtsºÍµÍãÐÖµtw½øÐÐ¹éÒ»»¯£¬·½±ãÖ®ºó¼ÆËãmasksºÍ¼â·åµÄÖÐÐÄÊ±¼ä
compute_masks():¶ÔÓÚÃ¿Ò»¸öÌáÈ¡µ½µÄwave£¬¼ÆËãÆämasksµÄÖµ
*/
/*******************************************************copy the components to the wave**************************************************************/
/****************************************************normalize²Ù×÷*************************************************************/
/****************************************************compute_masks²Ù×÷*************************************************************/
__global__ void normalize(float *nor_ary, float *flit_ary,float tw,float ts, size_t N)
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < N)
{
if (flit_ary[tid] >= ts) nor_ary[tid] = 1;
else if (nor_ary[tid] < tw) nor_ary[tid] = 0;
else nor_ary[tid] = (flit_ary[tid] - tw) / (ts - tw);
}
}