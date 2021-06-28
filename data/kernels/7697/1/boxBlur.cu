#include "includes.h"
__global__ void boxBlur(uchar4 *d_img, uchar4 *d_blurred, int numRows, int numCols, int k){


printf("Hilo: %d\n", threadIdx.x);
printf("KERNEL PAANIC R: %d, G: %d, B: %d\n", d_img[numCols*69+45].x, d_img[numCols*1 +7].y, d_img[numCols*1 +7].z);

int id = threadIdx.x;
int accumulator, cornerX, cornerY, sumRGB[] = {0,0,0};


for(int x=id; x<numRows; x+=blockDim.x){
//printf("hilo: %d, col: %d, blockDim.x: %d\n", id, x, blockDim.x);

for(int y=0; y<numCols; y++){
//printf("hilo: %d, row: %d, col: %d, blockDim.x: %d\n", id, x, y, blockDim.x);
//printf("(%d,%d,%d) R: %d, G: %d, B: %d\n", id, x, y, d_img[numCols*x+y].x, d_img[numCols*x+y].y, d_img[numCols*x+y].z);
//cout << y << endl;

int pixel = numCols * x + y;
uchar4 pixelColor = d_img[pixel];
//printf("R: %d, G: %d, B: %d\n", pixelColor.x, pixelColor.y, pixelColor.z);
sumRGB[0]=0;
sumRGB[1]=0;
sumRGB[2]=0;
accumulator=0;
if(k%2==0){
cornerX = x-(k-1)/2;
cornerY = y-(k-1)/2;
}else{
cornerX = x-(k-2)/2;
cornerY = y-(k-2)/2;
}

for(int i = 0; i < k; i++){
for(int j = 0; j < k; j++){
if(cornerX+i>=0 && cornerX+i<numRows && cornerY+j>=0 && cornerY+j<numCols) {
int index = numCols * (cornerX+i) + (cornerY+j);
uchar4 color = d_img[index];
sumRGB[0] += (int) color.x;
sumRGB[1] += (int) color.y;
sumRGB[2] += (int) color.z;
accumulator++;
}
}
}



if(accumulator == 0){
printf("WHAAAA?\n" );
sumRGB[0] = pixelColor.x;
sumRGB[1] = pixelColor.y;
sumRGB[2] = pixelColor.z;
}else{
sumRGB[0] = sumRGB[0] / accumulator;
sumRGB[1] = sumRGB[1] / accumulator;
sumRGB[2] = sumRGB[2] / accumulator;
}

uchar4 color;
color.x = sumRGB[0];
color.y = sumRGB[1];
color.z = sumRGB[2];

//myLock.lock();
d_blurred[pixel] = color;
//myLock.unlock();
//printf("Original R: %d, G: %d, B: %d\n", pixelColor.x, pixelColor.y, pixelColor.z);
//printf("Blurred  R: %d, G: %d, B: %d\n", d_blurred[pixel].x, d_blurred[pixel].y, d_blurred[pixel].z);

}

}


}