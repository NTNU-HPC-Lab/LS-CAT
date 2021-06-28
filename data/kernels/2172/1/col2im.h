#ifndef COL2IM_H
#define COL2IM_H
 #include <darknet_cl/darknet.h>
void col2im_cpu(float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_im);

#ifdef GPU
void col2im_gpu(CLArray data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, CLArray data_im);
#endif
#endif
