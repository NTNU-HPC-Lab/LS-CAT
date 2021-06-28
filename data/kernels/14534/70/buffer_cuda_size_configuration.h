/*
 *  Copyright 2011-2013 Maxim Milakov
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <memory>

namespace nnforge
{
	namespace cuda
	{
		class buffer_cuda_size_configuration
		{
		public:
			buffer_cuda_size_configuration();

			void add_constant_buffer(size_t buffer_size);

			void add_per_entry_buffer(size_t buffer_size);

			void add_per_entry_linear_addressing_through_texture(unsigned int tex_per_entry);

			size_t constant_buffer_size;
			size_t per_entry_buffer_size;
			size_t max_entry_buffer_size;
			unsigned int max_tex_per_entry;
			unsigned int buffer_count;
		};
	}
}
