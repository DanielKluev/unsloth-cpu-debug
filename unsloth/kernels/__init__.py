# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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

from ..device_type import IS_CPU_DEBUG

if not IS_CPU_DEBUG:
    from .cross_entropy_loss import (
        fast_cross_entropy_loss,
        post_patch_loss_function,
        patch_loss_functions,
    )
    from .rms_layernorm import (
        fast_rms_layernorm,
        patch_rms_layernorm,
        unpatch_rms_layernorm,
    )
    from .layernorm import (
        fast_layernorm,
        patch_layernorm,
    )
    from .rope_embedding import fast_rope_embedding, inplace_rope_embedding
    from .swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel
    from .geglu import (
        geglu_exact_forward_kernel,
        geglu_exact_backward_kernel,
        geglu_approx_forward_kernel,
        geglu_approx_backward_kernel,
    )
    from .fast_lora import (
        get_lora_parameters,
        get_lora_parameters_bias,
        apply_lora_mlp_swiglu,
        apply_lora_mlp_geglu_exact,
        apply_lora_mlp_geglu_approx,
        apply_lora_qkv,
        apply_lora_o,
        fast_lora_forward,
    )
    from .fp8 import *  # This step is to ensure that we patch the FbgmemFP8Linear and FP8Linear's forward functions before the execution of model creation so that this applies to compiled non fast inference models as well
    from .utils import (
        fast_dequantize,
        fast_gemv,
        QUANT_STATE,
        fast_linear_forward,
        matmul_lora,
    )

    from .flex_attention import (
        HAS_FLEX_ATTENTION,
        slow_attention_softcapping,
        slow_inference_attention_softcapping,
        create_flex_attention_causal_mask,
        create_flex_attention_sliding_window_mask,
    )
else:
    # CPU debug stubs - these functions are not usable on CPU
    # but allow the rest of the codebase to import without errors
    def _cpu_stub(*args, **kwargs):
        raise RuntimeError("This kernel function requires a GPU and is not available in CPU debug mode.")

    fast_cross_entropy_loss = _cpu_stub
    post_patch_loss_function = _cpu_stub
    patch_loss_functions = _cpu_stub
    fast_rms_layernorm = _cpu_stub
    patch_rms_layernorm = _cpu_stub
    unpatch_rms_layernorm = _cpu_stub
    fast_layernorm = _cpu_stub
    patch_layernorm = _cpu_stub
    fast_rope_embedding = _cpu_stub
    inplace_rope_embedding = _cpu_stub
    swiglu_fg_kernel = _cpu_stub
    swiglu_DWf_DW_dfg_kernel = _cpu_stub
    geglu_exact_forward_kernel = _cpu_stub
    geglu_exact_backward_kernel = _cpu_stub
    geglu_approx_forward_kernel = _cpu_stub
    geglu_approx_backward_kernel = _cpu_stub
    get_lora_parameters = _cpu_stub
    get_lora_parameters_bias = _cpu_stub
    apply_lora_mlp_swiglu = _cpu_stub
    apply_lora_mlp_geglu_exact = _cpu_stub
    apply_lora_mlp_geglu_approx = _cpu_stub
    apply_lora_qkv = _cpu_stub
    apply_lora_o = _cpu_stub
    fast_lora_forward = _cpu_stub
    fast_dequantize = _cpu_stub
    fast_gemv = _cpu_stub
    QUANT_STATE = 0
    fast_linear_forward = _cpu_stub
    matmul_lora = _cpu_stub
    HAS_FLEX_ATTENTION = False
    slow_attention_softcapping = _cpu_stub
    slow_inference_attention_softcapping = _cpu_stub
    create_flex_attention_causal_mask = _cpu_stub
    create_flex_attention_sliding_window_mask = _cpu_stub

import os

if "UNSLOTH_ZOO_IS_PRESENT" not in os.environ:
    try:
        if IS_CPU_DEBUG:
            print(
                "🦥 Unsloth: Running in CPU debug mode - GPU kernels are disabled."
            )
        else:
            print(
                "🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning."
            )
    except:
        print("Unsloth: Will patch your computer to enable 2x faster free finetuning.")
del os
