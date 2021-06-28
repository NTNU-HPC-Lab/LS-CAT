#ifndef IMAGE_H
#define IMAGE_H

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include "box.h"
#include "darknet.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef OPENCV
//DARKNET_API image CALLBACK ipl_to_image(IplImage* src);
//DARKNET_API void CALLBACK ipl_into_image(IplImage* src, image im);
void *open_video_stream(const char *f, int c, int w, int h, int fps);
image get_image_from_stream(void *p);
image load_image_cv(char *filename, int channels);
int show_image_cv(image im, const char* name, int ms);
#endif

DARKNET_API float CALLBACK get_pixel(image m, int x, int y, int c);
DARKNET_API float CALLBACK get_pixel_extend(image m, int x, int y, int c);
DARKNET_API void CALLBACK set_pixel(image m, int x, int y, int c, float val);
DARKNET_API float CALLBACK get_color(int c, int x, int max);
void write_label(image a, int r, int c, image *characters, char *string, float *rgb);
image image_distance(image a, image b);
void scale_image(image m, float s);
image rotate_crop_image(image im, float rad, float s, int w, int h, float dx, float dy, float aspect);
image random_crop_image(image im, int w, int h);
DARKNET_API image CALLBACK random_augment_image(image im, float angle, float aspect, int low, int high, int w, int h);
augment_args random_augment_args(image im, float angle, float aspect, int low, int high, int w, int h);
DARKNET_API void CALLBACK letterbox_image_into(image im, int w, int h, image boxed);
DARKNET_API image CALLBACK resize_max(image im, int max);
void translate_image(image m, float s);
DARKNET_API void CALLBACK embed_image(image source, image dest, int dx, int dy);
void place_image(image im, int w, int h, int dx, int dy, image canvas);
void saturate_image(image im, float sat);
void exposure_image(image im, float sat);
DARKNET_API void CALLBACK distort_image(image im, float hue, float sat, float val);
void saturate_exposure_image(image im, float sat, float exposure);
void rgb_to_hsv(image im);
void hsv_to_rgb(image im);
void yuv_to_rgb(image im);
void rgb_to_yuv(image im);


DARKNET_API image CALLBACK collapse_image_layers(image source, int border);
image collapse_images_horz(image *ims, int n);
DARKNET_API image CALLBACK collapse_images_vert(image *ims, int n);

void print_image(image m);

DARKNET_API image CALLBACK make_empty_image(int w, int h, int c);
void copy_image_into(image src, image dest);

DARKNET_API image CALLBACK get_image_layer(image m, int l);

#ifdef __cplusplus
}
#endif

#endif

