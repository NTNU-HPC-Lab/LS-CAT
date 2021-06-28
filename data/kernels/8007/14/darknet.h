#ifndef __DARKNET_H__
#define __DARKNET_H__
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#ifndef HAVE_STRUCT_TIMESPEC
#define HAVE_STRUCT_TIMESPEC 1
#endif
#ifndef HAVE_SIGNAL_H
#define HAVE_SIGNAL_H 1
#endif
#endif
#include <pthread.h>

#include <time.h>

#ifdef _WIN32
#ifdef DLL_EXPORT
#define DARKNET_API __declspec(dllexport)
#else
#define DARKNET_API __declspec(dllimport)
#endif
#ifndef CALLBACK
#define CALLBACK __stdcall
#endif
#else
#define DARKNET_API
#define CALLBACK
#endif

#define SECRET_NUM -1234
DARKNET_API extern int gpu_index;

#ifdef GPU
    #define BLOCK 512

    #include "cuda_runtime.h"
    #include "curand.h"
    #include "cublas_v2.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct{
    int classes;
    char **names;
} metadata;

metadata get_metadata(char *file);

typedef struct{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;
DARKNET_API tree * CALLBACK read_tree(char *filename);

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;

typedef enum{
    PNG, BMP, TGA, JPG
} IMTYPE;

typedef enum{
    MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    ISEG,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK
} LAYER_TYPE;

typedef enum{
    SSE, MASKED, L1, SEG, SMOOTH,WGAN
} COST_TYPE;

typedef struct{
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
} update_args;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void (*forward)   (struct layer, struct network);
    void (*backward)  (struct layer, struct network);
    void (*update)    (struct layer, update_args);
    void (*forward_gpu)   (struct layer, struct network);
    void (*backward_gpu)  (struct layer, struct network);
    void (*update_gpu)    (struct layer, update_args);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontsave;
    int dontloadscales;
    int numload;

    float temperature;
    float probability;
    float scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    int   * counts;
    float ** sums;
    float * rand;
    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float * concat;
    float * concat_delta;

    float * binary_weights;

    float * biases;
    float * bias_updates;

    float * scales;
    float * scale_updates;

    float * weights;
    float * weight_updates;

    float * delta;
    float * output;
    float * loss;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;
    
    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *dc_cpu; 

    float * binary_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;

    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

    tree *softmax_tree;

    size_t workspace_size;

#ifdef GPU
    int *indexes_gpu;

    float *z_gpu;
    float *r_gpu;
    float *h_gpu;

    float *temp_gpu;
    float *temp2_gpu;
    float *temp3_gpu;

    float *dh_gpu;
    float *hh_gpu;
    float *prev_cell_gpu;
    float *cell_gpu;
    float *f_gpu;
    float *i_gpu;
    float *g_gpu;
    float *o_gpu;
    float *c_gpu;
    float *dc_gpu; 

    float *m_gpu;
    float *v_gpu;
    float *bias_m_gpu;
    float *scale_m_gpu;
    float *bias_v_gpu;
    float *scale_v_gpu;

    float * combine_gpu;
    float * combine_delta_gpu;

    float * prev_state_gpu;
    float * forgot_state_gpu;
    float * forgot_delta_gpu;
    float * state_gpu;
    float * state_delta_gpu;
    float * gate_gpu;
    float * gate_delta_gpu;
    float * save_gpu;
    float * save_delta_gpu;
    float * concat_gpu;
    float * concat_delta_gpu;

    float * binary_input_gpu;
    float * binary_weights_gpu;

    float * mean_gpu;
    float * variance_gpu;

    float * rolling_mean_gpu;
    float * rolling_variance_gpu;

    float * variance_delta_gpu;
    float * mean_delta_gpu;

    float * x_gpu;
    float * x_norm_gpu;
    float * weights_gpu;
    float * weight_updates_gpu;
    float * weight_change_gpu;

#ifdef CUDNN_HALF
    float * weights_gpu16;
    float * weight_updates_gpu16;
#endif

    float * biases_gpu;
    float * bias_updates_gpu;
    float * bias_change_gpu;

    float * scales_gpu;
    float * scale_updates_gpu;
    float * scale_change_gpu;

    float * output_gpu;
    float * loss_gpu;
    float * delta_gpu;
    float * rand_gpu;
    float * squared_gpu;
    float * norms_gpu;
#ifdef CUDNN
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnTensorDescriptor_t normTensorDesc;
    cudnnTensorDescriptor_t normDstTensorDesc;
#ifdef CUDNN_HALF
    cudnnTensorDescriptor_t normDstTensorDescF16;
#endif
    cudnnFilterDescriptor_t weightDesc;
    cudnnFilterDescriptor_t dweightDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo;
    cudnnConvolutionBwdDataAlgo_t bd_algo;
    cudnnConvolutionBwdFilterAlgo_t bf_algo;
#endif
#endif
};

void free_layer(layer);

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network{
    int n;
    int batch;
    size_t *seen;
    int *t;
    float epoch;
    int subdivisions;
    layer *layers;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    int gpu_index;
    tree *hierarchy;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;
    float clip;

#ifdef GPU
    float *input_gpu;
    float *truth_gpu;
#ifdef CUDNN_HALF
    float **input16_gpu;
    float **output16_gpu;
    size_t *max_input16_size;
    size_t *max_output16_size;
#endif
    float *delta_gpu;
    float *output_gpu;
#endif

} network;

typedef struct {
    int w;
    int h;
    float scale;
    float rad;
    float dx;
    float dy;
    float aspect;
} augment_args;

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;


typedef struct{
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA, ISEG_DATA
} data_type;

typedef struct load_args{
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int c;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int center;
    int coords;
    float jitter;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    data *d;
    image *im;
    image *resized;
    data_type type;
    tree *hierarchy;
} load_args;

typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;


DARKNET_API network * CALLBACK load_network(char *cfg, char *weights, int clear);
DARKNET_API load_args CALLBACK get_base_args(network *net);

DARKNET_API void CALLBACK free_data(data d);

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

DARKNET_API pthread_t CALLBACK load_data(load_args args);
DARKNET_API list * CALLBACK read_data_cfg(char *filename);
list *read_cfg(char *filename);
DARKNET_API unsigned char * CALLBACK read_file(char *filename);
DARKNET_API data CALLBACK resize_data(data orig, int w, int h, int c);
DARKNET_API data * CALLBACK tile_data(data orig, int divs, int size, int c);
DARKNET_API data CALLBACK select_data(data *orig, int *inds);

DARKNET_API void CALLBACK forward_network(network *net);
DARKNET_API void CALLBACK backward_network(network *net);
DARKNET_API void CALLBACK update_network(network *net);


DARKNET_API float CALLBACK dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
DARKNET_API void CALLBACK axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
DARKNET_API void CALLBACK copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
DARKNET_API void CALLBACK scal_cpu(int N, float ALPHA, float *X, int INCX);
DARKNET_API void CALLBACK fill_cpu(int N, float ALPHA, float * X, int INCX);
DARKNET_API void CALLBACK normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
DARKNET_API void CALLBACK softmax(float *input, int n, float temp, int stride, float *output);

int best_3d_shift_r(image a, image b, int min, int max);
#ifdef GPU
DARKNET_API void CALLBACK axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
DARKNET_API void CALLBACK fill_gpu(int N, float ALPHA, float * X, int INCX);
DARKNET_API void CALLBACK scal_gpu(int N, float ALPHA, float * X, int INCX);
DARKNET_API void CALLBACK copy_gpu(int N, float * X, int INCX, float * Y, int INCY);

DARKNET_API void CALLBACK cuda_set_device(int n);
DARKNET_API void CALLBACK cuda_free(float *x_gpu);
DARKNET_API float * CALLBACK cuda_make_array(float *x, size_t n);
DARKNET_API void CALLBACK cuda_pull_array(float *x_gpu, float *x, size_t n);
DARKNET_API float CALLBACK cuda_mag_array(float *x_gpu, size_t n);
DARKNET_API void CALLBACK cuda_push_array(float *x_gpu, float *x, size_t n);

DARKNET_API void CALLBACK forward_network_gpu(network *net);
DARKNET_API void CALLBACK backward_network_gpu(network *net);
DARKNET_API void CALLBACK update_network_gpu(network *net);

DARKNET_API float CALLBACK train_networks(network **nets, int n, data d, int interval);
DARKNET_API void CALLBACK sync_nets(network **nets, int n, int interval);
DARKNET_API void CALLBACK harmless_update_network_gpu(network *net);
#endif
image get_label(image **characters, char *string, int size);
DARKNET_API void CALLBACK save_image(image im, const char *name);
DARKNET_API void CALLBACK save_image_options(image im, const char *name, IMTYPE f, int quality);
DARKNET_API void CALLBACK get_next_batch(data d, int n, int offset, float *X, float *y);
DARKNET_API void CALLBACK grayscale_image_3c(image im);
DARKNET_API void CALLBACK normalize_image(image p);
DARKNET_API void CALLBACK normalize_image2(image p);
DARKNET_API void CALLBACK matrix_to_csv(matrix m);
DARKNET_API float CALLBACK train_network_sgd(network *net, data d, int n);
DARKNET_API void CALLBACK rgbgr_image(image im);
DARKNET_API data CALLBACK copy_data(data d);
DARKNET_API data CALLBACK concat_data(data d1, data d2);
DARKNET_API data CALLBACK load_cifar10_data(char *filename);
DARKNET_API float CALLBACK matrix_topk_accuracy(matrix truth, matrix guess, int k);
DARKNET_API void CALLBACK matrix_add_matrix(matrix from, matrix to);
DARKNET_API void CALLBACK scale_matrix(matrix m, float scale);
DARKNET_API matrix CALLBACK csv_to_matrix(char *filename);
DARKNET_API float * CALLBACK network_accuracies(network *net, data d, int n);
DARKNET_API float CALLBACK train_network_datum(network *net);
DARKNET_API image CALLBACK make_random_image(int w, int h, int c);

DARKNET_API void CALLBACK denormalize_connected_layer(layer l);
DARKNET_API void CALLBACK denormalize_convolutional_layer(layer l);
DARKNET_API void CALLBACK statistics_connected_layer(layer l);
DARKNET_API void CALLBACK rescale_weights(layer l, float scale, float trans);
DARKNET_API void CALLBACK rgbgr_weights(layer l);
DARKNET_API image * CALLBACK get_weights(layer l);

void get_detection_detections(layer l, int w, int h, float thresh, detection *dets);

DARKNET_API char * CALLBACK option_find_str(list *l, char *key, char *def);
DARKNET_API int CALLBACK option_find_int(list *l, char *key, int def);
DARKNET_API int CALLBACK option_find_int_quiet(list *l, char *key, int def);

DARKNET_API network * CALLBACK parse_network_cfg(char *filename);
DARKNET_API void CALLBACK save_weights(network *net, char *filename);
DARKNET_API void CALLBACK load_weights(network *net, char *filename);
DARKNET_API void CALLBACK save_weights_upto(network *net, char *filename, int cutoff);
DARKNET_API void CALLBACK load_weights_upto(network *net, char *filename, int start, int cutoff);

DARKNET_API void CALLBACK zero_objectness(layer l);
DARKNET_API void CALLBACK get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
DARKNET_API int CALLBACK get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
DARKNET_API void CALLBACK free_network(network *net);
DARKNET_API void CALLBACK set_batch_network(network *net, int b);
DARKNET_API void CALLBACK fuse_conv_batchnorm(network* net);
void set_temp_network(network *net, float t);
DARKNET_API image CALLBACK load_image(char *filename, int w, int h, int c);
DARKNET_API image CALLBACK load_image_color(char *filename, int w, int h);
DARKNET_API image CALLBACK make_image(int w, int h, int c);
DARKNET_API image CALLBACK resize_image(image im, int w, int h);
DARKNET_API void CALLBACK censor_image(image im, int dx, int dy, int w, int h);
DARKNET_API image CALLBACK letterbox_image(image im, int w, int h);
DARKNET_API image CALLBACK crop_image(image im, int dx, int dy, int w, int h);
DARKNET_API image CALLBACK center_crop_image(image im, int w, int h);
DARKNET_API image CALLBACK resize_min(image im, int min);
DARKNET_API image CALLBACK resize_max(image im, int max);
DARKNET_API image CALLBACK threshold_image(image im, float thresh);
DARKNET_API image CALLBACK mask_to_rgb(image mask);
DARKNET_API int CALLBACK resize_network(network *net, int w, int h);
DARKNET_API void CALLBACK free_matrix(matrix m);
DARKNET_API image CALLBACK copy_image(image p);
DARKNET_API float CALLBACK get_current_rate(network *net);
DARKNET_API void CALLBACK composite_3d(char *f1, char *f2, char *out, int delta);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h, int c);
DARKNET_API size_t CALLBACK get_current_batch(network *net);
DARKNET_API void CALLBACK constrain_image(image im);
DARKNET_API image CALLBACK get_network_image_layer(network *net, int i);
DARKNET_API layer CALLBACK get_network_output_layer(network *net);
DARKNET_API void CALLBACK top_predictions(network *net, int n, int *index);
DARKNET_API void CALLBACK flip_image(image a);
DARKNET_API image CALLBACK float_to_image(int w, int h, int c, float *data);
DARKNET_API void CALLBACK ghost_image(image source, image dest, int dx, int dy);
float network_accuracy(network *net, data d);
DARKNET_API void CALLBACK random_distort_image(image im, float hue, float saturation, float exposure);
DARKNET_API void CALLBACK fill_image(image m, float s);
DARKNET_API image CALLBACK grayscale_image(image im);
DARKNET_API void CALLBACK rotate_image_cw(image im, int times);
DARKNET_API double CALLBACK what_time_is_it_now();
DARKNET_API image CALLBACK rotate_image(image m, float rad);
DARKNET_API float CALLBACK box_iou(box a, box b);
DARKNET_API data CALLBACK load_all_cifar10();
DARKNET_API box_label * CALLBACK read_boxes(char *filename, int *n);
box float_to_box(float *f, int stride);

DARKNET_API matrix CALLBACK network_predict_data(network *net, data test);
DARKNET_API image CALLBACK get_network_image(network *net);
DARKNET_API float* CALLBACK network_predict(network *net, float *input);

int network_width(network *net);
int network_height(network *net);
float *network_predict_image(network *net, image im);
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets);
DARKNET_API detection* CALLBACK get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
DARKNET_API void CALLBACK free_detections(detection *dets, int n);

DARKNET_API void CALLBACK reset_network_state(network *net, int b);

DARKNET_API char ** CALLBACK get_labels(char *filename);
DARKNET_API void CALLBACK do_nms_obj(detection *dets, int total, int classes, float thresh);
DARKNET_API void CALLBACK do_nms_sort(detection *dets, int total, int classes, float thresh);

DARKNET_API matrix CALLBACK make_matrix(int rows, int cols);

DARKNET_API void CALLBACK free_image(image m);
DARKNET_API float CALLBACK train_network(network *net, data d);
DARKNET_API pthread_t CALLBACK load_data_in_thread(load_args args);
void load_data_blocking(load_args args);
DARKNET_API list * CALLBACK get_paths(char *filename);
DARKNET_API void CALLBACK hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride);
DARKNET_API void CALLBACK change_leaves(tree *t, char *leaf_list);

DARKNET_API int CALLBACK find_int_arg(int argc, char **argv, char *arg, int def);
DARKNET_API float CALLBACK find_float_arg(int argc, char **argv, char *arg, float def);
DARKNET_API int CALLBACK find_arg(int argc, char* argv[], char *arg);
DARKNET_API char * CALLBACK find_char_arg(int argc, char **argv, char *arg, char *def);
DARKNET_API char * CALLBACK basecfg(char *cfgfile);
DARKNET_API void CALLBACK find_replace(char *str, char *orig, char *rep, char *output);
DARKNET_API void CALLBACK free_ptrs(void **ptrs, int n);
DARKNET_API char * CALLBACK fgetl(FILE *fp);
DARKNET_API void CALLBACK strip(char *s);
DARKNET_API float CALLBACK sec(clock_t clocks);
DARKNET_API void ** CALLBACK list_to_array(list *l);
DARKNET_API void CALLBACK top_k(float *a, int n, int k, int *index);
DARKNET_API int * CALLBACK read_map(char *filename);
DARKNET_API void CALLBACK error(const char *s);
DARKNET_API int CALLBACK max_index(float *a, int n);
DARKNET_API int CALLBACK max_int_index(int *a, int n);
DARKNET_API int CALLBACK sample_array(float *a, int n);
DARKNET_API int * CALLBACK random_index_order(int min, int max);
DARKNET_API void CALLBACK free_list(list *l);
float mse_array(float *a, int n);
DARKNET_API float CALLBACK variance_array(float *a, int n);
DARKNET_API float CALLBACK mag_array(float *a, int n);
DARKNET_API void CALLBACK scale_array(float *a, int n, float s);
DARKNET_API float CALLBACK mean_array(float *a, int n);
DARKNET_API float CALLBACK sum_array(float *a, int n);
DARKNET_API void CALLBACK normalize_array(float *a, int n);
DARKNET_API int * CALLBACK read_intlist(char *s, int *n, int d);
DARKNET_API size_t CALLBACK rand_size_t();
DARKNET_API float CALLBACK rand_normal();
DARKNET_API float CALLBACK rand_uniform(float min, float max);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif
