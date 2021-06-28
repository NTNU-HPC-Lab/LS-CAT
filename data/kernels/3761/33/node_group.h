/*
Copyright (C) 2020 Bruno Golosio
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef NODEGROUPH
#define NODEGROUPH

#define MAX_N_NODE_GROUPS 128

struct NodeGroupStruct
{
  int node_type_;
  int i_node_0_;
  int n_node_;
  int n_port_;
  int n_param_;
  double *get_spike_array_;
  int *spike_count_;
  float *rec_spike_times_;
  int *n_rec_spike_times_;
  int max_n_rec_spike_times_;
  float *den_delay_arr_;
};

#endif

