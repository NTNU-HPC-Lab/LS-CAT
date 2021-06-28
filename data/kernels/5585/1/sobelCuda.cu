#include "includes.h"
/*
***** sobel.cpp *****

Description:
Author:       John M. Weiss, Ph.D.
Modified By:  Jeremy Goens
Class:        CSC 461 Programming Languages
Date:         Fall 2017
Compilation   g++ -std=c++11 -g *.cpp
*/

using namespace std;

typedef unsigned char byte;

unsigned greyScale(char* str, byte*& image, byte*& pixels, unsigned int& width, unsigned int& height);
unsigned writeImage(byte*& image, unsigned int& width, unsigned int& height, char* str);
void sobel(byte*& image, byte*& edged, unsigned int& width, unsigned int& height);
void sobelOpenMP(byte*& image, byte*& edged, unsigned int& width, unsigned int& height);


/*
Description:
Author:       John M. Weiss, Ph.D.
Modified By:  Jeremy Goens
Class:        CSC 461 Programming Languages
Date:         Fall 2017
Compilation   g++ *.cpp
*/
__global__ void sobelCuda(byte* image, byte* edged, int width, int height){
int x = threadIdx.x + blockIdx.x * blockDim.x;

int j = x/width;
int i = x%width;

if( i < 1 || i >= (width-1) || j < 1 || j >= (height-1) )
return;

int gX = (-1)*image[(i-1)+((j-1)*width)];
gX += (-2)*image[(i)+((j-1)*width)];
gX += (-1)*image[(i+1)+((j-1)*width)];
gX += 1*image[(i-1)+((j+1)*width)];
gX += 2*image[(i)+((j+1)*width)];
gX += 1*image[(i+1)+((j+1)*width)];

int gY = (-1)*image[(i-1)+((j-1)*width)];
gY += 1*image[(i+1)+((j-1)*width)];
gY += (-2)*image[(i-1)+((j)*width)];
gY += 2*image[(i+1)+((j)*width)];
gY += (-1)*image[(i-1)+((j+1)*width)];
gY += 1*image[(i+1)+((j+1)*width)];

edged[i+(j*width)] = ( byte )min( sqrt( (float) (gX*gX)+(gY*gY)), 255.0);

//Black Edges all around
}