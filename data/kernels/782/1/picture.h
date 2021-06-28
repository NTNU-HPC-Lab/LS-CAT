#ifndef PICTURE_H
#define PICTURE_H

// Color at index 0 is top left, at index 0 is top left + 1 to the right
// All colors are in [0, 255]
struct Picture {
  unsigned width, height;
  unsigned char *colors;
};

// Sets the picture at location "picture" into a random picture of dimensions
// width and height
void setRandomPicture(struct Picture *picture, unsigned width,
  unsigned height);

void printPicture(struct Picture *picture);

// Temporary turning for demo purposes
void turnPictureParallel(struct Picture *in, struct Picture *out,
  double radians);

__global__ void turnPictureKernel(unsigned char *d_in, unsigned char *d_out,
  unsigned picWidth, unsigned picHeight, double sn, double cs);

#endif
