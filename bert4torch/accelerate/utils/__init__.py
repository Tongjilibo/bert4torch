# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .dataclasses import CustomDtype
from .environment import (
    parse_flag_from_env,
    patch_environment,
    str_to_bool,
)
from .environment import (
    check_cuda_p2p_ib_support,
    get_gpu_info,
    parse_flag_from_env,
    patch_environment,
    str_to_bool,
)
from .imports import (
    is_bnb_available,
    is_cuda_available,
    is_deepspeed_available,
    is_hpu_available,
    is_mlu_available,
    is_mps_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_torch_xla_available,
    is_xpu_available,
    is_neuron_available
)
from .modeling import (
    check_device_map,
    check_tied_parameters_in_config,
    check_tied_parameters_on_same_device,
    compute_module_sizes,
    convert_file_size_to_int,
    dtype_byte_size,
    find_tied_parameters,
    get_balanced_memory,
    get_max_layer_size,
    get_max_memory,
    infer_auto_device_map,
    named_module_tensors,
    retie_parameters,
    set_module_tensor_to_device,
)
from .offload import (
    OffloadedWeightsLoader,
    PrefixedDataset,
    extract_submodules_state_dict,
    load_offloaded_weight,
    offload_state_dict,
    offload_weight,
    save_offload_index,
)
from .versions import compare_versions, is_torch_version
from .operations import (
    find_device,
    honor_type,
    is_namedtuple,
    is_torch_tensor,
    send_to_device,
)
from .other import (
    recursive_getattr,
)
