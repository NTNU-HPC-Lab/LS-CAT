#ifndef SCIMAGE_H
#define SCIMAGE_H

extern "C" {
#include "libavcodec/avcodec.h"
}

#include "cuda.h"
#include "cuda_runtime.h"

class ScGPUImage
{
public:
    explicit ScGPUImage();
    ~ScGPUImage();

    enum enScPixFormat {
        ScPixFormat_Unknown,
        ScPixFormat_YUV420P,
        ScPixFormat_RGBA
    };

    static enScPixFormat getScPixFormat(AVPixelFormat format);
    bool copyFromAVFrame(AVFrame *frame);
    void resize(int pixel_w, int pixel_h, enScPixFormat format);

    static const int DEFAULT_IMAGE_ALIGNMENT = 128;

    /* The actual image may just stored at left-top corner of the total image memory */
    int m_pixel_width;      /* image width on pixel unit */
    int m_pixel_height;     /* image height on pixel unit */
    int m_data_width;        /* image width stored on byte unit */
    int m_data_height;      /* image height stored on byte unit */
    enScPixFormat m_pix_fmt;
    dev_t *m_dev_addr;
    cudaArray_t m_array_addr;

    void *m_glContext;
    unsigned int m_pbo;
    unsigned int m_time_stamp;
    cudaGraphicsResource_t m_resource;
    cudaSurfaceObject_t    m_cuda_surface;

    int m_align;
};

#endif // SCIMAGE_H
