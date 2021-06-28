#include "includes.h"
__global__ void prepare_boundary_accel_on_device(const float * d_accel, float * d_send_accel_buffer, const int num_interfaces, const int max_nibool_interfaces, const int * d_nibool_interfaces, const int * d_ibool_interfaces){
int id;
int iglob;
int iloc;
int iinterface;
id = threadIdx.x + (blockIdx.x) * (blockDim.x) + ((gridDim.x) * (blockDim.x)) * (threadIdx.y + (blockIdx.y) * (blockDim.y));
for (iinterface = 0; iinterface <= num_interfaces - (1); iinterface += 1) {
if (id < d_nibool_interfaces[iinterface]) {
iloc = id + (max_nibool_interfaces) * (iinterface);
iglob = d_ibool_interfaces[iloc] - (1);
d_send_accel_buffer[(iloc) * (3) + 0] = d_accel[(iglob) * (3) + 0];
d_send_accel_buffer[(iloc) * (3) + 1] = d_accel[(iglob) * (3) + 1];
d_send_accel_buffer[(iloc) * (3) + 2] = d_accel[(iglob) * (3) + 2];
}
}
}