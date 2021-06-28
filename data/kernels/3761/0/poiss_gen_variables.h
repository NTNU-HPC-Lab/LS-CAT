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

#ifndef POISSGENVARIABLESH
#define POISSGENVARIABLESH

#include <string>

enum {
  i_rate = 0,
  i_origin,
  i_start,
  i_stop,
  N_POISS_GEN_SCAL_PARAM
};

const std::string poiss_gen_scal_param_name[N_POISS_GEN_SCAL_PARAM] = {
  "rate",
  "origin",
  "start",
  "stop",
};

#define rate param[i_rate]
#define origin param[i_origin]
#define start param[i_start]
#define stop param[i_stop]

#endif
