#include "includes.h"
__device__ void conv(const float* gm, float* convolved, int bh, int bw, int ih, int iw, int ch, int cw, int smH, int smW, int k, float* sm, int gID, int tID, int nT, int rel_row, int rel_col, int nRows, int stopPrefetchRowID, int lastActiveThreadID) {

for(int i=k; i<=nRows; i++)
{
/*
----prefetch a pixel value from GM and store it in register----

all threads fetch the cell value immediately below to the current cell iteratively

last thread in the block would fetch k cells immediately below the current cell

boundary check would be needed for the blocks that act on the bottom most partition of the input image to prevent it from prefetching out of image values.
*/
float reg;
float regArr[K];
if(i <= stopPrefetchRowID){
reg = gm[i * iw + gID];
if(tID == lastActiveThreadID){
for(int j=1; j<=k-1; j++){
regArr[j] = gm[(i * iw) + gID + j];
}
}
}
// load k * k pixels above the current cell
float imgPixels[K*K];
for(int r=i-k; r<i; r++){
for(int c=0; c<k; c++){
/* translate the indices to [0,k] using r - (i-k) as imgPixels is of size k*k */
imgPixels[(r-i+k)*k + c] = sm[r * smW + tID + c];
}
}
/*multiply image pixel values with filter values (direct convolution) */
float convolvedCell = 0.0;
for(int c=0; c<k*k; c++){
convolvedCell += cm[c]*imgPixels[c];
}
//place the convolvedCell value into convolvedMatrix
int cID = ( ( (rel_row * bh) + (i-k) ) * cw )+( rel_col * nT )+tID;
if(cID < 0 || cID >= ch*cw ) {
printf("cID : %d, tID : %d, gID : %d\n", cID, tID, gID );
}
convolved[cID] = convolvedCell;
__syncthreads();
if(i <= stopPrefetchRowID){
sm[i * smW + tID] = reg;
if(tID == lastActiveThreadID){
for(int j=1; j<=k-1; j++){
int sID = i *smW + tID + j;
sm[sID] = regArr[j];
}
}
}
__syncthreads();
}


}
__global__ void conv_kernel(const float* gm, float* convolved, int bh, int bw, int ih, int iw, int ch, int cw, int smH, int smW, int k) {

int tID = threadIdx.x;
int bID = blockIdx.x;
int nT = blockDim.x;
int nB = gridDim.x;
int nBx = iw / nT;
//printf("num of blocks is %d\n", nB);
//printf("nB in a row is %d\n", nBx);
//check for right border or bottom border thread block
bool isBottomBorder = false;
bool isRightBorder = false;
// bottom border thread block
if(bID >= nB - nBx) {
//printf("bID : %d is bottom border\n", bID);
isBottomBorder = true;
}
// right border thread block
if((bID+1) % nBx == 0){
//printf("bID : %d is right border\n", bID);
isRightBorder = true;
}

// ---------------- Load k rows from GM into SM ----------------------

__shared__ float sm[ (BLCH + K - 1) * (BLCW + K - 1) ];
// rel_row and rel_col maps the Thread Block to appropriate position
int rel_row = bID / nBx;
int rel_col = bID % nBx;
// (rel_row * bh * iw) covers all the cells before row_ids bh, 2bh, 3bh ..
// gID finally maps threads to cells at rows 0, bh, 2bh, 3bh, ...
int gID = (rel_row * bh * iw) + (rel_col * nT) + tID;

for(int i=0; i<k; i++){

int sID = i * smW + tID;
sm[sID] = gm[i * iw + gID];
/* if last thread in the block, it should fetch additional k-1 pixels
in each row which are needed for computation of the convolution
*/
if(!isRightBorder && tID == nT-1){
for(int j=1; j<=k-1; j++){
sID = (i * smW) + tID + j;
sm[sID] = gm[i * iw + gID + j];
}
}

}

__syncthreads();

if( !isBottomBorder && !isRightBorder ){
int lastActiveThreadID = nT - 1;
int nRows = bh + k - 1;
int stopPrefetchRowID = nRows;
conv( gm, convolved, bh, bw,
ih, iw, ch, cw, smH, smW, k,
sm, gID, tID, nT, rel_row, rel_col,
nRows, stopPrefetchRowID, lastActiveThreadID );
}
else if( isBottomBorder && isRightBorder ){
/* make the last k-1 threads in the block to be idle. as there is no convolution needed for them */
if(tID < (nT - (k-1))){
int nRows = bh;
int stopPrefetchRowID = nRows - 1;
int lastActiveThreadID = nT - k;
conv( gm, convolved, bh, bw,
ih, iw, ch, cw, smH, smW, k,
sm, gID, tID, nT, rel_row, rel_col,
nRows, stopPrefetchRowID, lastActiveThreadID );
}
}
else if( isBottomBorder ){
int nRows = bh;
int stopPrefetchRowID = nRows-1;
int lastActiveThreadID = nT - 1;
conv( gm, convolved, bh, bw,
ih, iw, ch, cw, smH, smW, k,
sm, gID, tID, nT, rel_row, rel_col,
nRows, stopPrefetchRowID, lastActiveThreadID );


}
else if( isRightBorder ){
/* make the last k-1 threads in the block to be idle. as there is no convolution needed for them */
if(tID < (nT - (k-1))){
int nRows = bh + k - 1;
int stopPrefetchRowID = nRows;
int lastActiveThreadID = nT - k;
conv( gm, convolved, bh, bw,
ih, iw, ch, cw, smH, smW, k,
sm, gID, tID, nT, rel_row, rel_col,
nRows, stopPrefetchRowID, lastActiveThreadID );
}

}



}