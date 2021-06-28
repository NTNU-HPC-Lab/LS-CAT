#include "includes.h"
__global__ void render_final(float *points3d_polar, float * selection, float * depth_render, int * img,  int * render, int oh, int ow)
{
int x = blockIdx.x * TILE_DIM + threadIdx.x;
int y = blockIdx.y * TILE_DIM + threadIdx.y;
int w = gridDim.x * TILE_DIM;
int h = w /2;
int maxsize = oh * ow;

for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
{

int iw = x;
int ih = y + j;


int tx = round((points3d_polar[(ih * w + iw) * 3 + 1] + 1)/2 * ow - 0.5);
int ty = round((points3d_polar[(ih * w + iw) * 3 + 2] + 1)/2 * oh - 0.5);

float tx_offset = ((points3d_polar[(ih * w + iw) * 3 + 1] + 1)/2 * ow - 0.5);
float ty_offset = ((points3d_polar[(ih * w + iw) * 3 + 2] + 1)/2 * oh - 0.5);

float tx00 = 0;
float ty00 = 0;

float tx01 = ((points3d_polar[(ih * w + iw + 1) * 3 + 1] + 1)/2 * ow - 0.5) - tx_offset;
float ty01 = ((points3d_polar[(ih * w + iw + 1) * 3 + 2] + 1)/2 * oh - 0.5) - ty_offset;

float tx10 = ((points3d_polar[((ih + 1) * w + iw) * 3 + 1] + 1)/2 * ow - 0.5) - tx_offset;
float ty10 = ((points3d_polar[((ih + 1) * w + iw) * 3 + 2] + 1)/2 * oh - 0.5) - ty_offset;

float tx11 = ((points3d_polar[((ih+1) * w + iw + 1) * 3 + 1] + 1)/2 * ow - 0.5) - tx_offset;
float ty11 = ((points3d_polar[((ih+1) * w + iw + 1) * 3 + 2] + 1)/2 * oh - 0.5) - ty_offset;

float t00 = 0 * (float)tx00 + (float)tx01 * -1.0/3  + (float)tx10 *  2.0/3   + (float)tx11 *  1.0/3;
float t01 = 0 * (float)ty00 + (float)ty01 * -1.0/3  + (float)ty10 *  2.0/3   + (float)ty11 *  1.0/3;
float t10 = 0 * (float)tx00 + (float)tx01 *  2.0/3  + (float)tx10 * -1.0/3   + (float)tx11 *  1.0/3;
float t11 = 0 * (float)ty00 + (float)ty01 *  2.0/3  + (float)ty10 * -1.0/3   + (float)ty11 *  1.0/3;

float det = t00 * t11 - t01 * t10 + 1e-10;

//printf("%f %f %f %f %f\n", t00, t01, t10, t11, det);

float it00, it01, it10, it11;

it00 = t11/det;
it01 = -t01/det;
it10 = -t10/det;
it11 = t00/det;

//printf("inverse %f %f %f %f\n", it00, it01, it10, it11);

int this_depth = (int)(12800/128 * points3d_polar[(ih * w + iw) * 3 + 0]);
int delta00 = (int)(12800/128 * points3d_polar[(ih * w + iw) * 3 + 0]) - (int)(100 * depth_render[(ty * ow + tx)]);
int delta01 = (int)(12800/128 * points3d_polar[(ih * w + iw + 1) * 3 + 0]) - (int)(100 * depth_render[(ty * ow + tx + 1)]);
int delta10 = (int)(12800/128 * points3d_polar[((ih + 1) * w + iw) * 3 + 0]) - (int)(100 * depth_render[((ty+1) * ow + tx)]);
int delta11 = (int)(12800/128 * points3d_polar[((ih+1) * w + iw + 1) * 3 + 0]) - (int)(100 * depth_render[((ty+1) * ow + tx + 1)]);

int mindelta = min(min(delta00, delta01), min(delta10, delta11));
int maxdelta = max(max(delta00, delta01), max(delta10, delta11));

int depth00 = (int)(12800/128 * points3d_polar[(ih * w + iw) * 3 + 0]);
int depth01 = (int)(12800/128 * points3d_polar[(ih * w + iw + 1) * 3 + 0]);
int depth10 = (int)(12800/128 * points3d_polar[((ih+1) * w + iw) * 3 + 0]);
int depth11 = (int)(12800/128 * points3d_polar[((ih+1) * w + iw+1) * 3 + 0]);
int max_depth =  max(max(depth00, depth10), max(depth01, depth11));
int min_depth =  min(min(depth00, depth10), min(depth01, depth11));
int delta_depth = max_depth - min_depth;

int txmin = floor(tx_offset + min(min(tx00, tx11), min(tx01, tx10)));
int txmax = ceil(tx_offset + max(max(tx00, tx11), max(tx01, tx10)));
int tymin = floor(ty_offset + min(min(ty00, ty11), min(ty01, ty10)));
int tymax = ceil(ty_offset + max(max(ty00, ty11), max(ty01, ty10)));

float newx, newy;
int r,g,b;
int itx, ity;

//render[(ty * ow + tx)] = img[ih * w + iw];
//selection[(ty * ow + tx)] = 1.0;

float tolerance = 0.1 * this_depth > 10? 0.1 * this_depth : 10;
float tolerance2 = 0.05 * max_depth > 10? 0.05 * max_depth: 10;

float flank = 0.01;
if ((delta_depth < tolerance2) && (y > 1 * h/8) && (y < (h*7)/8))
if (((mindelta > - tolerance) && (maxdelta <  tolerance)) && (this_depth < 10000)) {
if (((txmax - txmin) * (tymax - tymin) < 1600) && (txmax - txmin < 40) && (tymax - tymin < 40))
{
for (itx = txmin; itx < txmax; itx ++)
for (ity = tymin; ity < tymax; ity ++)
{ if (( 0 <= itx) && (itx < ow) && ( 0 <= ity) && (ity < oh))
{
newx = (itx - tx_offset) * it00 + it10 * (ity - ty_offset);
newy = (itx - tx_offset) * it01 + it11 * (ity - ty_offset);

//printf("%f %f\n", newx, newy);
if ((newx > -flank) && (newx < 1 + flank) && (newy > -flank) && (newy < 1 + flank))
{
if (newx < 0) newx = 0;
if (newy < 0) newy = 0;
if (newx > 1) newx = 1;
if (newy > 1) newy = 1;

r = img[(ih * w + iw)] / (256*256) * (1-newx) * (1-newy) + img[(ih * w + iw + 1)] / (256*256) * (1-newx) * (newy) + img[((ih+1) * w + iw)] / (256*256) * (newx) * (1-newy) + img[((ih+1) * w + iw + 1)] / (256*256) * newx * newy;
g = img[(ih * w + iw)] / 256 % 256 * (1-newx) * (1-newy) + img[(ih * w + iw + 1)] / 256 % 256 * (1-newx) * (newy) + img[((ih+1) * w + iw)] / 256 % 256  * (newx) * (1-newy)  + img[((ih+1) * w + iw + 1)] / 256 % 256 * newx * newy;
b = img[(ih * w + iw)] % 256 * (1-newx) * (1-newy) + img[(ih * w + iw + 1)] % 256 * (1-newx) * (newy) + img[((ih+1) * w + iw)] % 256 * (newx) * (1-newy)  + img[((ih+1) * w + iw + 1)] % 256 * newx * newy ;

if (r > 255) r = 255;
if (g > 255) g = 255;
if (b > 255) b = 255;

if ((ity * ow + itx > 0) && (ity * ow + itx < maxsize)) {
render[(ity * ow + itx)] = r * 256 * 256 + g * 256 + b;
selection[(ity * ow + itx)] = 1.0 / abs(det);
}
}
}
}

}
}

}



}