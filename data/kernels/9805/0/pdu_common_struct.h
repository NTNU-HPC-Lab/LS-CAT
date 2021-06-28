//
// Created by psaw on 9/17/16.
//

#include <stdint.h>
#include <string.h>
//#include "log4c.h"

#ifndef SMPP_CONST
#define SMPP_CONST
//#include "smpp_constant.h"
#endif

#ifndef PDU_COMMON_STRUCT
#define PDU_COMMON_STRUCT

//extern log4c_category_t* statLogger;
//extern log4c_category_t* decoderLogger;

typedef struct byte_buffer_context_struct {
    uint8_t *buffer;
    uint64_t readIndex;
    uint64_t limit;
    uint64_t initialReadIndex;
} ByteBufferContext;

typedef struct smpp_header_struct {
    uint32_t commandLength;
    uint32_t commandId;
    uint32_t commandStatus;
    uint32_t sequenceNumber;
} SmppHeader;

#endif

