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
    from .llama import FastLlamaModel
    from .loader import FastLanguageModel, FastVisionModel, FastTextModel, FastModel
    from .mistral import FastMistralModel
    from .qwen2 import FastQwen2Model
    from .qwen3 import FastQwen3Model
    from .qwen3_moe import FastQwen3MoeModel
    from .granite import FastGraniteModel
    from .sentence_transformer import FastSentenceTransformer

    try:
        from .falcon_h1 import FastFalconH1Model
    except:
        # transformers_version < 4.53.0 does not have falcon_h1 so silently skip it for now
        pass
    from .dpo import PatchDPOTrainer, PatchKTOTrainer
    from ._utils import is_bfloat16_supported, is_vLLM_available, __version__
    from .rl import PatchFastRL, vLLMSamplingParams
else:
    from .._version import __version__

    def is_bfloat16_supported():
        return False

    def is_vLLM_available():
        return False

    # CPU debug stubs for model classes - importable but raise on GPU operations
    _CPU_MSG = (
        "Unsloth: `{}` requires a GPU. Running in CPU debug mode.\n"
        "Tokenizer loading, dataset preparation and chat templates work on CPU.\n"
        "Model loading, training and saving require a GPU."
    )

    class _CpuStubModel:
        """Stub model returned by from_pretrained() in CPU debug mode."""
        def __init__(self):
            self.config = {}

        def __repr__(self):
            return "<CpuStubModel: GPU required for real model>"

    class _CpuModelStub:
        """Base stub for model classes in CPU debug mode."""
        @staticmethod
        def from_pretrained(
            model_name = None,
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
            load_in_8bit = False,
            load_in_16bit = False,
            full_finetuning = False,
            token = None,
            device_map = "sequential",
            rope_scaling = None,
            fix_tokenizer = True,
            trust_remote_code = False,
            use_gradient_checkpointing = "unsloth",
            resize_model_vocab = None,
            revision = None,
            use_exact_model_name = False,
            offload_embedding = False,
            float32_mixed_precision = None,
            fast_inference = False,
            gpu_memory_utilization = 0.5,
            float8_kv_cache = False,
            random_state = 3407,
            max_lora_rank = 64,
            disable_log_stats = True,
            qat_scheme = None,
            load_in_fp8 = False,
            unsloth_tiled_mlp = False,
            *args,
            **kwargs,
        ):
            from transformers import AutoTokenizer
            print(
                "Unsloth: Running in CPU debug mode.\n"
                "Loading tokenizer only - model is a stub.\n"
                "GPU is required for actual model loading."
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token = token,
                trust_remote_code = trust_remote_code,
            )
            return _CpuStubModel(), tokenizer

        @staticmethod
        def get_peft_model(model, *args, **kwargs):
            print(
                "Unsloth: CPU debug mode - get_peft_model is a no-op.\n"
                "Returning stub model unchanged."
            )
            return model

        @staticmethod
        def pre_patch(*args, **kwargs):
            raise RuntimeError(_CPU_MSG.format("pre_patch"))

        @staticmethod
        def for_training(model = None, *args, **kwargs):
            raise RuntimeError(_CPU_MSG.format("for_training"))

        @staticmethod
        def for_inference(model = None, *args, **kwargs):
            raise RuntimeError(_CPU_MSG.format("for_inference"))

    class FastLlamaModel(_CpuModelStub): pass
    class FastLanguageModel(_CpuModelStub): pass
    class FastVisionModel(_CpuModelStub): pass
    class FastTextModel(_CpuModelStub): pass
    class FastModel(_CpuModelStub): pass
    class FastMistralModel(_CpuModelStub): pass
    class FastQwen2Model(_CpuModelStub): pass
    class FastQwen3Model(_CpuModelStub): pass
    class FastQwen3MoeModel(_CpuModelStub): pass
    class FastGraniteModel(_CpuModelStub): pass
    class FastSentenceTransformer(_CpuModelStub): pass
    class FastFalconH1Model(_CpuModelStub): pass

    def PatchDPOTrainer(*args, **kwargs):
        raise RuntimeError(_CPU_MSG.format("PatchDPOTrainer"))

    def PatchKTOTrainer(*args, **kwargs):
        raise RuntimeError(_CPU_MSG.format("PatchKTOTrainer"))

    class PatchFastRL:
        pass

    class vLLMSamplingParams:
        pass
