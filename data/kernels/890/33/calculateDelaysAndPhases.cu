#include "includes.h"
__global__ void calculateDelaysAndPhases(double * gpuDelays, double lo, double sampletime, int fftsamples, int fftchannels, int samplegranularity, float * rotationPhaseInfo, int *sampleShifts, float* fractionalSampleDelays)
{
size_t ifft = threadIdx.x + blockIdx.x * blockDim.x;
size_t iant = blockIdx.y;
int numffts = blockDim.x * gridDim.x;
double meandelay, deltadelay, netdelaysamples_f, startphase;
double d0, d1, d2, a, b;
double * interpolator = &(gpuDelays[iant*4]);
double filestartoffset = gpuDelays[iant*4+3];
float fractionaldelay;
int netdelaysamples;

// evaluate the delay for the given FFT of the given antenna

// calculate values at the beginning, middle, and end of this FFT
d0 = interpolator[0]*ifft*ifft + interpolator[1]*ifft + interpolator[2];
d1 = interpolator[0]*(ifft+0.5)*(ifft+0.5) + interpolator[1]*(ifft+0.5) + interpolator[2];
d2 = interpolator[0]*(ifft+1.0)*(ifft+1.0) + interpolator[1]*(ifft+1.0) + interpolator[2];

// use these to calculate a linear interpolator across the FFT, as well as a mean value
a = d2-d0; //this is the delay gradient across this FFT
b = d0 + (d1 - (a*0.5 + d0))/3.0; //this is the delay at the start of the FFT
meandelay = a*0.5 + b; //this is the delay in the middle of the FFT
deltadelay = a / fftsamples; // this is the change in delay per sample across this FFT window

netdelaysamples_f = (meandelay - filestartoffset) / sampletime;
netdelaysamples = __double2int_rn(netdelaysamples_f/samplegranularity) * samplegranularity;

// Save the integer number of sample shifts
sampleShifts[iant*numffts + ifft] = netdelaysamples;

// Save the fractional delay
fractionaldelay = (float)(-(netdelaysamples_f - netdelaysamples)*2*M_PI/fftsamples);  // radians per FFT channel
fractionalSampleDelays[iant*numffts + ifft] = fractionaldelay;

// set the fringe rotation phase for the first sample of a given FFT of a given antenna
startphase = b*lo;
rotationPhaseInfo[iant*numffts*2 + ifft*2] = (float)(startphase - int(startphase))*2*M_PI;
rotationPhaseInfo[iant*numffts*2 + ifft*2 + 1] = (float)(deltadelay * lo)*2*M_PI;
}