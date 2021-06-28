#include <stdalign.h>
#include "pdu_common_struct.h"

#ifndef SMPP_PDU_STRUCT_CUDA
#define SMPP_PDU_STRUCT_CUDA
#define MAX_CORRELATION_LENGTH 20

#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

typedef struct cuda_pdu_context_direct_struct {
    uint32_t start;
    uint32_t length;
    char correlationId[MAX_CORRELATION_LENGTH];
} CudaPduContext;

typedef struct cuda_tlv_struct {
    uint16_t tag;
    uint32_t length;
    uint8_t value[10];
    char tagName[10];
} CudaTlv;

typedef struct cuda_address_struct {
    int8_t ton;
    int8_t npi;
    char addressValue[21];
} CudaAddress;

typedef struct cuda_base_sm_req_struct {
    SmppHeader header;
    char serviceType[6];
    CudaAddress sourceAddress;
    CudaAddress destinationAddress;
    uint8_t esmClass;
    uint8_t protocolId;
    uint8_t priority;
    char scheduleDeliveryTime[17];
    char validityPeriod[17];
    uint8_t registeredDelivery;
    uint8_t replaceIfPresent;
    uint8_t dataCoding;
    uint8_t defaultMsgId;
    uint8_t smLength;
    uint8_t shortMessage[256];
    uint16_t tlvCount;
    CudaTlv tlvList[10];
} CudaBaseSmReq;

typedef struct cuda_decoded_context_struct {
    char correlationId[MAX_CORRELATION_LENGTH];
    uint32_t commandId;
    CudaBaseSmReq pduStruct;
} CudaDecodedContext;

typedef struct cuda_dim_struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} CudaDim;

typedef struct cuda_metadata {
    int length;
    uint8_t *pduBuffer;
    CudaPduContext *cudaPduContexts;
    CudaDecodedContext *decodedPduStructList;
    uint64_t pduBufferLength;
    CudaDim blockDim;
    CudaDim gridDim;
    uint8_t isTunerMode;
} CudaMetadata;

void decodeCuda(CudaMetadata cudaMetadata);

void decodeCudaDynamic(CudaMetadata cudaMetadata);

void initCudaParameters(uint32_t pduContextSize, uint64_t pduBufferLength);

CudaPduContext *allocatePinnedPduContext(int length);

void freePinndedPduContext(int length, CudaPduContext *pduContexts);
void freePinndedMemory();

void cudaTest();

#endif