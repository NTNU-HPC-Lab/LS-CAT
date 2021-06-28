#include "includes.h"
__global__ void sobelEdgeDetectionSharedMem2(int *input, int *output, int width, int height, int thresh) {

int regArr[4][4];

int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
int j = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

if ( i > 0 && j > 0 && i < width - 1 && j < height - 1)
{

regArr[0][0] = input[width * (j-1) + i - 1];
regArr[0][1] = input[width * (j-1) + i    ];
regArr[0][2] = input[width * (j-1) + i + 1];
regArr[0][3] = input[width * (j-1) + i + 2];
regArr[1][0] = input[width * (j)   + i - 1];
regArr[1][1] = input[width * (j)   + i    ];
regArr[1][2] = input[width * (j)   + i + 1];
regArr[1][3] = input[width * (j)   + i + 2];
regArr[2][0] = input[width * (j+1) + i - 1];
regArr[2][1] = input[width * (j+1) + i    ];
regArr[2][2] = input[width * (j+1) + i + 1];
regArr[2][3] = input[width * (j+1) + i + 2];
regArr[3][0] = input[width * (j+2) + i - 1];
regArr[3][1] = input[width * (j+2) + i    ];
regArr[3][2] = input[width * (j+2) + i + 1];
regArr[3][3] = input[width * (j+2) + i + 2];

__syncthreads();


int sum1 = 0, sum2 = 0, magnitude;
int num = 3;

for(int xind = 1; xind < num; xind++)
{
for(int yind = 1; yind < num; yind++)
{
sum1 = regArr[xind+1][yind-1] -     regArr[xind-1][yind-1]
+ 2 * regArr[xind+1][yind  ] - 2 * regArr[xind-1][yind  ]
+     regArr[xind+1][yind+1] -     regArr[xind-1][yind+1];

sum2 = regArr[xind-1][yind-1] + 2 * regArr[xind][yind-1] + regArr[xind+1][yind-1]
- regArr[xind-1][yind+1] - 2 * regArr[xind][yind+1] - regArr[xind+1][yind+1];

magnitude = sum1 * sum1 + sum2 * sum2;

if(magnitude > thresh)
output[(j + yind - 1) * width + (i + xind - 1)] = 255;
else
output[(j + yind - 1) * width + (i + xind - 1)] = 0;

}
}
}
}