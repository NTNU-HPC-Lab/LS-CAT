#include "includes.h"
__global__ void Overlay_Cuda( int x_position, int y_position, unsigned char* main, int main_linesize, unsigned char* overlay, int overlay_linesize, int overlay_w, int overlay_h, unsigned char* overlay_alpha, int alpha_linesize, int alpha_adj_x, int alpha_adj_y)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if (x >= overlay_w + x_position ||
y >= overlay_h + y_position ||
x < x_position ||
y < y_position ) {

return;
}

int overlay_x = x - x_position;
int overlay_y = y - y_position;

float alpha = 1.0;
if (alpha_linesize) {
alpha = overlay_alpha[alpha_adj_x * overlay_x  + alpha_adj_y * overlay_y * alpha_linesize] / 255.0f;
}

main[x + y*main_linesize] = alpha * overlay[overlay_x + overlay_y * overlay_linesize] + (1.0f - alpha) * main[x + y*main_linesize];
}