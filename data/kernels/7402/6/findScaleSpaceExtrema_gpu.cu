#include "includes.h"
__device__ void addpoint(){

}
__device__ unsigned int d_PointCounter[1]; //choose 60 suport 16384 pixel size image (log2(16384) - 2)*5
__device__ float *pd[60];
//choose 72 suport 16384 pixel size image (log2(16384) - 2)*6
__device__ float *pgpyr[72];
__device__ int temsize;
//36 suppose the max Octave is 12
__constant__ int d_oIndex[36];

static const int BLOCK_SIZE_ONE_DIM = 32;
static const int Descript_BLOCK_SIZE = 32;

__global__ void test_gpu(int pitch,int height)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if(y<height)
{
pgpyr[0][y*pitch+x] += 100;
}
if(y*pitch+x<5)
printf("%f\n",pd[0][y*pitch+x]);
}
__global__ void findScaleSpaceExtrema_gpu(float *d_point,int p_pitch,int s, int width ,int pitch ,int height,const int threshold,const int nOctaveLayers,const int maxNum){

int x = blockIdx.x*blockDim.x+threadIdx.x;
int y = blockIdx.y*blockDim.y+threadIdx.y;

//avoid extract the unstable border points
if(y >= height - SIFT_IMG_BORDER || x >= width - SIFT_IMG_BORDER || x<SIFT_IMG_BORDER || y<SIFT_IMG_BORDER)
return;

float *currptr = pd[s]  +y*pitch+x;
float *prevptr = pd[s-1]+y*pitch+x;
float *nextptr = pd[s+1]+y*pitch+x;

int o = s/(nOctaveLayers+2);
float val = *currptr;
int step = pitch;
int c = 0;
if( std::abs(val) > threshold &&
((val > 0 && val >= currptr[c-1] && val >= currptr[c+1] &&
val >= currptr[c-step-1] && val >= currptr[c-step] && val >= currptr[c-step+1] &&
val >= currptr[c+step-1] && val >= currptr[c+step] && val >= currptr[c+step+1] &&
val >= nextptr[c] && val >= nextptr[c-1] && val >= nextptr[c+1] &&
val >= nextptr[c-step-1] && val >= nextptr[c-step] && val >= nextptr[c-step+1] &&
val >= nextptr[c+step-1] && val >= nextptr[c+step] && val >= nextptr[c+step+1] &&
val >= prevptr[c] && val >= prevptr[c-1] && val >= prevptr[c+1] &&
val >= prevptr[c-step-1] && val >= prevptr[c-step] && val >= prevptr[c-step+1] &&
val >= prevptr[c+step-1] && val >= prevptr[c+step] && val >= prevptr[c+step+1]) ||
(val < 0 && val <= currptr[c-1] && val <= currptr[c+1] &&
val <= currptr[c-step-1] && val <= currptr[c-step] && val <= currptr[c-step+1] &&
val <= currptr[c+step-1] && val <= currptr[c+step] && val <= currptr[c+step+1] &&
val <= nextptr[c] && val <= nextptr[c-1] && val <= nextptr[c+1] &&
val <= nextptr[c-step-1] && val <= nextptr[c-step] && val <= nextptr[c-step+1] &&
val <= nextptr[c+step-1] && val <= nextptr[c+step] && val <= nextptr[c+step+1] &&
val <= prevptr[c] && val <= prevptr[c-1] && val <= prevptr[c+1] &&
val <= prevptr[c-step-1] && val <= prevptr[c-step] && val <= prevptr[c-step+1] &&
val <= prevptr[c+step-1] && val <= prevptr[c+step] && val <= prevptr[c+step+1])))
{
/*adjustLocalExtrema*/
const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);
const float deriv_scale = img_scale*0.5f;
const float second_deriv_scale = img_scale;
const float cross_deriv_scale = img_scale*0.25f;
float Vs=0, Vx=0, Vy=0, contr=0;
float dx,dy,ds,dxx,dyy,dxy;
int j = 0,layer;
//get the x,y,s,Vs,Vx,Vy or return
for( ; j < SIFT_MAX_INTERP_STEPS; j++ )
{
currptr = pd[s]  +y*pitch+x;
prevptr = pd[s-1]+y*pitch+x;
nextptr = pd[s+1]+y*pitch+x;

//the first derivative of x,y and scale
dx = (currptr[1] - currptr[-1])*deriv_scale;
dy = (currptr[pitch] - currptr[-pitch])*deriv_scale;;
ds = (nextptr[0] - prevptr[0])*deriv_scale;
float v2 = currptr[0]*2;

//the second derivative of x,y,scale
dxx = (currptr[1] + currptr[-1] - v2)*second_deriv_scale;
dyy = (currptr[pitch] + currptr[-pitch] - v2)*second_deriv_scale;
float dss = (nextptr[0] + prevptr[0] - v2)*second_deriv_scale;
dxy = (currptr[pitch+1] - currptr[1-pitch] -
currptr[-1+pitch] + currptr[-pitch-1])*cross_deriv_scale;
float dxs = (nextptr[1] - nextptr[-1] -
prevptr[1] + prevptr[-1])*cross_deriv_scale;
float dys = (nextptr[pitch] - nextptr[-pitch] -
prevptr[pitch] + prevptr[-pitch])*cross_deriv_scale;

//Algebraic cousin
float idxx = dyy*dss - dys*dys;
float idxy = dys*dxs - dxy*dss;
float idxs = dxy*dys - dyy*dxs;
//idet is the det,the matrix's determinant countdown
float idet = __fdividef(1.0f, idxx*dxx + idxy*dxy + idxs*dxs);
float idyy = dxx*dss - dxs*dxs;
float idys = dxy*dxs - dxx*dys;
float idss = dxx*dyy - dxy*dxy;
////////////////////////
///  A(dxx, dxy, dxs,
///    dxy, dyy, dys,
///    dxs, dys, dss);
///
///  A*(idxx, idxy, idxs,
///     idxy, idyy, idys,
///     idxs, idys, idss);
///
///  B(dx,dy,dz)
/////////////////////////
//dX = (A^-1)*B
float pdx = idet*(idxx*dx + idxy*dy + idxs*ds);
float pdy = idet*(idxy*dx + idyy*dy + idys*ds);
float pds = idet*(idxs*dx + idys*dy + idss*ds);

Vx = -pdx;
Vy = -pdy;
Vs = -pds;

//because of the judgment is before the updated value,so
//this iteration final get the x,y,s(intger) and the Vx,Vy,Vz(<0.5).
//The accurate extrema location is x+Vx,y+Vy.

if( std::abs(Vs) < 0.5f && std::abs(Vx) < 0.5f && std::abs(Vy) < 0.5f )
break;

//get nearest intger
x += int(Vx > 0 ? ( Vx + 0.5 ) : (Vx - 0.5));
y += int(Vy > 0 ? ( Vy + 0.5 ) : (Vy - 0.5));
s += int(Vs > 0 ? ( Vs + 0.5 ) : (Vs - 0.5));

layer = s - o*(nOctaveLayers+2);

if( layer < 1 || layer > nOctaveLayers ||
y < SIFT_IMG_BORDER || y >= height - SIFT_IMG_BORDER  ||
x < SIFT_IMG_BORDER || x >= width - SIFT_IMG_BORDER )
return;

}//for
if( j >= SIFT_MAX_INTERP_STEPS )
return;

//After the iterative,get the x,y,s,(Vx,Vy,Vs)(<0.5).

{
//remove the small energy points which essily influenced by image noise
float t = dx*Vx + dy*Vy + ds*Vs;
contr = currptr[0]*img_scale + t * 0.5f;
if( std::abs( contr ) * nOctaveLayers < 0.04 )
return;

// principal curvatures are computed using the trace and det of Hessian
float tr = dxx + dyy;
float det = dxx*dyy-dxy*dxy;

if( det <= 0 || tr*tr*10 >= (10 + 1)*(10 + 1)*det )
return;
}

layer = s - o*(nOctaveLayers+2);

//addpoint;
unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
idx = (idx>=maxNum ? maxNum-1 : idx);
//        d_point[idx*KEYPOINTS_SIZE] = (x + Vx)*(1 << o);
//        d_point[idx*KEYPOINTS_SIZE+1] = (y + Vy)*(1 << o);
//        d_point[idx*KEYPOINTS_SIZE+2] = o + (s<<8) + ((int)(((Vs + 0.5)*255)+0.5) << 16);
//        float size = 1.6*__powf(2.f, (layer + Vs) / nOctaveLayers)*(1 << o)*2;
//        d_point[idx*KEYPOINTS_SIZE+3] = size;
//        d_point[idx*KEYPOINTS_SIZE+4] = std::abs(contr);
//        d_point[idx*KEYPOINTS_SIZE+6] = s;
//        d_point[idx*KEYPOINTS_SIZE+7] = x;
//        d_point[idx*KEYPOINTS_SIZE+8] = y;

d_point[idx] = (x + Vx)*(1 << o);
d_point[idx+p_pitch*1] = (y + Vy)*(1 << o);
float oct_lay1 =o + (layer<<8) + ((int)(((Vs + 0.5)*255)+0.5) << 16);
//        if(idx == 0)
//            printf("o = %d ,Vs = %f ,oct_lay1 = %f ,x = %f , y = %f\n",o,Vs,oct_lay1,d_point[idx],d_point[idx+p_pitch*1]);
//int oct_lay = oct_lay1;
d_point[idx+p_pitch*2] = oct_lay1;
float size = 1.6*__powf(2.f, (layer + Vs) / nOctaveLayers)*(1 << o)*2;
d_point[idx+p_pitch*3] = size;
d_point[idx+p_pitch*4] = std::abs(contr);
//        int _octave,_layer;
//        _octave = oct_lay & 255;
//        layer = (oct_lay >> 8) & 255;
//        _octave = _octave < 128 ? _octave : (-128 | _octave);
//        s = _octave*(nOctaveLayers+2)+layer;
//        x = round(d_point[idx]/(1<<_octave));
//        y = round(d_point[idx+p_pitch*1]/(1<<_octave));
//        d_point[idx+p_pitch*6] = s;
//        d_point[idx+p_pitch*7] = x;
//        d_point[idx+p_pitch*8] = y;

//temsize+=size*0.5f/(1 << o)*SIFT_ORI_RADIUS+0.5;
//        if(x<2000 && y<2000)
//            printf("%d,%d,%d\n",x,y,s);
//printf("%f \n",pd[0][100*2304+100]);

//        float scl_octv = size*0.5f/(1 << o);
//        //'+0.5' for rounding because scl_octv>0
//        int radius = SIFT_ORI_RADIUS * scl_octv+0.5;
//        //the procress of all point range, a square space.
//        int len = (radius*2+1)*(radius*2+1);
//        //int temBuffSize = len*4+2*SIFT_ORI_HIST_BINS+2;
//        atomicMax(&temsize,len);
}
}