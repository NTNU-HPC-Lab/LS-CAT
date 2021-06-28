#include "includes.h"
/*
* Copyright 2018 International Business Machines
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/



#define LOG_ERR(pid, fmt, x...) fprintf(stderr, "Process %d: " fmt, pid, ##x)
#define LOG_INF(pid, fmt, x...) printf("Process %d: " fmt, pid, ##x)

#define AFU_NAME "IBM,MEMCPY3"
#define AFU_MAX_PROCESSES 512

#define CACHELINESIZE	128
/* Queue sizes other than 512kB don't seem to work (still true?) */
#define QUEUE_SIZE	4095*CACHELINESIZE

#define MEMCPY_WED(queue, depth)			\
((((uint64_t)queue) & 0xfffffffffffff000ULL) |	\
(((uint64_t)depth) & 0xfffULL))

#define MEMCPY_WE_CMD(valid, cmd)		\
(((valid) & 0x1) |			\
(((cmd) & 0x3f) << 2))
#define MEMCPY_WE_CMD_VALID	(0x1 << 0)
#define MEMCPY_WE_CMD_WRAP	(0x1 << 1)
#define MEMCPY_WE_CMD_COPY		0
#define MEMCPY_WE_CMD_IRQ		1
#define MEMCPY_WE_CMD_STOP		2
#define MEMCPY_WE_CMD_WAKE_HOST_THREAD	3
#define MEMCPY_WE_CMD_INCREMENT		4
#define MEMCPY_WE_CMD_ATOMIC		5
#define MEMCPY_WE_CMD_TRANSLATE_TOUCH	6

/* global mmio registers */
#define MEMCPY_AFU_GLOBAL_CFG	0
#define MEMCPY_AFU_GLOBAL_TRACE	0x20

/* per-process mmio registers */
#define MEMCPY_AFU_PP_WED	0
#define MEMCPY_AFU_PP_STATUS	0x10
#define   MEMCPY_AFU_PP_STATUS_Terminated	0x8
#define   MEMCPY_AFU_PP_STATUS_Stopped		0x10

#define MEMCPY_AFU_PP_CTRL	0x18
#define   MEMCPY_AFU_PP_CTRL_Restart	(0x1 << 0)
#define   MEMCPY_AFU_PP_CTRL_Terminate	(0x1 << 1)
#define MEMCPY_AFU_PP_IRQ	0x28


struct memcpy_work_element {
volatile uint8_t cmd; /* valid, wrap, cmd */
volatile uint8_t status;
uint16_t length;
uint8_t cmd_extra;
uint8_t reserved[3];
uint64_t atomic_op;
uint64_t src;  /* also irq EA or atomic_op2 */
uint64_t dst;
} __packed;

struct memcpy_weq {
struct memcpy_work_element *queue;
struct memcpy_work_element *next;
struct memcpy_work_element *last;
int wrap;
int count;
};

struct memcpy_test_args {
int loop_count;
int size;
int irq;
int completion_timeout;
int reallocate;
char *device;
int wake_host_thread;
int increment;
int atomic_cas;
/* global vars */
int shmid;
char *lock;
char *counter;
};

__global__ void VecAdd(float* A, float* B, float* C, int N)
{
for (int i=0; i<N; i++) {
C[i] = A[i] + B[i];
}
}