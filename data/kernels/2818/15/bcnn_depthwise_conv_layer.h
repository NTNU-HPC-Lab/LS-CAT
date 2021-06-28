/*
 * Copyright (c) 2016-present Jean-Noel Braun..
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef BCNN_DEPTHWISE_CONV_LAYER_H
#define BCNN_DEPTHWISE_CONV_LAYER_H

#include "bcnn_net.h"
#include "bcnn_node.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bcnn_depthwise_conv_param {
    int num;
    int size;
    int stride;
    int pad;
    bcnn_activation activation;
    float *adam_m;
    float *adam_v;
#ifdef BCNN_USE_CUDA
    float *adam_m_gpu;
    float *adam_v_gpu;
#endif
} bcnn_depthwise_conv_param;

void bcnn_forward_depthwise_conv_layer(bcnn_net *net, bcnn_node *node);
void bcnn_backward_depthwise_conv_layer(bcnn_net *net, bcnn_node *node);
void bcnn_update_depthwise_conv_layer(bcnn_net *net, bcnn_node *node);
void bcnn_release_param_depthwise_conv_layer(bcnn_node *node);

#ifdef BCNN_USE_CUDA
void bcnn_forward_depthwise_conv_layer_gpu(bcnn_net *net, bcnn_node *node);
void bcnn_backward_depthwise_conv_layer_gpu(bcnn_net *net, bcnn_node *node);
#endif

#ifdef __cplusplus
}
#endif

#endif  // BCNN_DEPTHWISE_CONV_LAYER_H