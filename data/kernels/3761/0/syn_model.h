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

#ifndef SYNMODELH
#define SYNMODELH

#include <string>
#include <vector>

#define MAX_SYN_DT 16384
enum SynModels {
  i_null_syn_model = 0, i_test_syn_model, i_stdp_model,
  N_SYN_MODELS
};

const std::string syn_model_name[N_SYN_MODELS] = {
  "", "test_syn_model", "stdp"
};

class SynModel
{
 protected:
  int type_;
  int n_param_;
  const std::string *param_name_;
  float *d_param_arr_;
 public:
  virtual int Init() {return 0;}
  int GetNParam();
  std::vector<std::string> GetParamNames();
  bool IsParam(std::string param_name);
  int GetParamIdx(std::string param_name);
  virtual float GetParam(std::string param_name);
  virtual int SetParam(std::string param_name, float val);

  friend class NeuronGPU;
};

#endif
