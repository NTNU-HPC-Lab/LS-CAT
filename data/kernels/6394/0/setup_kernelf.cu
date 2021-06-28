#include "includes.h"



using namespace std;


float *valuesf;
float *weightf;
float maxWf;

float *matchf;
const int fSUMFLAG=0;
const int fKNAPSACKFLAG = 1;

const int fAVGFLAG=2;
const int fMATCHFLAG=3;
const int fINVERSESUMFLAG=4;

const int fMAXIMIZE=-1;
const int fMINIMIZE=1;




__global__ void setup_kernelf ( curandState *state, unsigned long seed )
{
curand_init ( seed, 0, 0, &state[0] );
}