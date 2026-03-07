"""
Tests for CPU debug mode.
Verifies that the codebase can be imported and used on CPU-only setups,
with GPU-dependent features properly stubbed out.
"""

import sys
import os
import tempfile

import pytest


@pytest.fixture(autouse = True)
def add_project_root():
    root = os.path.dirname(os.path.dirname(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)


class TestDeviceType:
    """Test device_type module on CPU."""

    def test_device_type_returns_cpu(self):
        from unsloth.device_type import DEVICE_TYPE
        assert DEVICE_TYPE == "cpu"

    def test_is_cpu_debug_true(self):
        from unsloth.device_type import IS_CPU_DEBUG
        assert IS_CPU_DEBUG is True

    def test_device_count_is_one(self):
        from unsloth.device_type import DEVICE_COUNT
        assert DEVICE_COUNT == 1

    def test_allow_prequantized_false(self):
        from unsloth.device_type import ALLOW_PREQUANTIZED_MODELS
        assert ALLOW_PREQUANTIZED_MODELS is False

    def test_allow_bitsandbytes_false(self):
        from unsloth.device_type import ALLOW_BITSANDBYTES
        assert ALLOW_BITSANDBYTES is False


class TestPackageImport:
    """Test that the main package imports without errors on CPU."""

    def test_unsloth_import(self):
        import unsloth
        assert hasattr(unsloth, "__version__")

    def test_version_string(self):
        import unsloth
        from unsloth._version import __version__
        assert unsloth.__version__ == __version__

    def test_dataprep_available(self):
        import unsloth
        assert hasattr(unsloth, "RawTextDataLoader")
        assert hasattr(unsloth, "TextPreprocessor")

    def test_fast_language_model_importable(self):
        from unsloth import FastLanguageModel
        assert FastLanguageModel is not None

    def test_fast_vision_model_importable(self):
        from unsloth import FastVisionModel
        assert FastVisionModel is not None

    def test_fast_text_model_importable(self):
        from unsloth import FastTextModel
        assert FastTextModel is not None

    def test_fast_model_importable(self):
        from unsloth import FastModel
        assert FastModel is not None

    def test_chat_template_importable(self):
        from unsloth import get_chat_template
        assert callable(get_chat_template)

    def test_trainer_importable(self):
        from unsloth import UnslothTrainer, UnslothTrainingArguments
        assert UnslothTrainer is not None
        assert UnslothTrainingArguments is not None


class TestModelStubBehavior:
    """Test that model stubs work correctly in CPU debug mode."""

    def test_from_pretrained_returns_stub_and_tokenizer(self):
        """from_pretrained should return a stub model and a real tokenizer."""
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM",
        )
        # Model is a stub
        assert "CpuStubModel" in repr(model)
        # Tokenizer is real
        assert hasattr(tokenizer, "encode")
        assert hasattr(tokenizer, "decode")

    def test_get_peft_model_noop(self):
        """get_peft_model should be a no-op on CPU, returning the model unchanged."""
        from unsloth import FastLanguageModel
        from unsloth.models import _CpuStubModel
        stub = _CpuStubModel()
        result = FastLanguageModel.get_peft_model(stub)
        assert result is stub

    def test_get_chat_template_functional(self):
        """get_chat_template should actually apply chat templates on CPU."""
        from unsloth import get_chat_template
        assert callable(get_chat_template)
        # It's the real function, not a stub
        import inspect
        sig = inspect.signature(get_chat_template)
        assert "chat_template" in sig.parameters

    def test_trainer_raises(self):
        from unsloth import UnslothTrainer
        with pytest.raises(RuntimeError, match = "CPU debug mode"):
            UnslothTrainer()

    def test_train_on_responses_only_functional(self):
        """train_on_responses_only is the real function on CPU, not a stub."""
        from unsloth import train_on_responses_only
        assert callable(train_on_responses_only)
        import inspect
        sig = inspect.signature(train_on_responses_only)
        assert "instruction_part" in sig.parameters
        assert "response_part" in sig.parameters
        assert "return_function" in sig.parameters

class TestKernelStubs:
    """Test that kernel stubs are properly provided on CPU."""

    def test_kernel_imports(self):
        from unsloth.kernels import (
            fast_cross_entropy_loss,
            fast_rms_layernorm,
            HAS_FLEX_ATTENTION,
            QUANT_STATE,
        )
        assert HAS_FLEX_ATTENTION is False
        assert QUANT_STATE == 0

    def test_kernel_stubs_raise_runtime_error(self):
        from unsloth.kernels import fast_cross_entropy_loss
        with pytest.raises(RuntimeError, match = "CPU debug mode"):
            fast_cross_entropy_loss()

    def test_all_kernel_stubs_callable(self):
        from unsloth.kernels import (
            fast_cross_entropy_loss,
            fast_rms_layernorm,
            fast_rope_embedding,
            fast_dequantize,
            fast_linear_forward,
            matmul_lora,
        )
        for fn in [fast_cross_entropy_loss, fast_rms_layernorm,
                    fast_rope_embedding, fast_dequantize,
                    fast_linear_forward, matmul_lora]:
            assert callable(fn)


class TestSaveStubs:
    """Test that save module stubs work on CPU."""

    def test_save_imports(self):
        from unsloth.save import patch_saving_functions
        assert callable(patch_saving_functions)

    def test_patch_saving_functions_noop(self):
        from unsloth.save import patch_saving_functions

        class MockModel:
            pass
        mock = MockModel()
        # Should return the model unchanged on CPU
        result = patch_saving_functions(mock)
        assert result is mock

    def test_save_stubs_raise(self):
        from unsloth.save import unsloth_save_model
        with pytest.raises(RuntimeError, match = "CPU debug mode"):
            unsloth_save_model()


class TestModelsStubs:
    """Test that model module stubs work on CPU."""

    def test_version(self):
        from unsloth.models import __version__
        from unsloth._version import __version__ as expected_version
        assert __version__ == expected_version

    def test_is_bfloat16_supported(self):
        from unsloth.models import is_bfloat16_supported
        assert is_bfloat16_supported() is False

    def test_is_vllm_available(self):
        from unsloth.models import is_vLLM_available
        assert is_vLLM_available() is False


class TestDataPrep:
    """Test that data preparation works on CPU."""

    def test_raw_text_loader(self):
        from unsloth.dataprep.raw_text import RawTextDataLoader

        class MockTokenizer:
            def __init__(self):
                self.eos_token = "</s>"
                self.eos_token_id = 2

            def __call__(self, text, return_tensors = None, add_special_tokens = False):
                words = text.split()
                token_ids = list(range(len(words)))
                if return_tensors == "pt":

                    class MockTensor:
                        def __init__(self, data):
                            self.data = data

                        def __getitem__(self, idx):
                            return self.data

                        def __len__(self):
                            return len(self.data)

                        def tolist(self):
                            return self.data

                    return {"input_ids": [MockTensor(token_ids)]}
                return {"input_ids": token_ids}

            def decode(self, token_ids, skip_special_tokens = False):
                return " ".join([f"word_{i}" for i in token_ids])

        tokenizer = MockTokenizer()
        loader = RawTextDataLoader(tokenizer, chunk_size = 5, stride = 2)

        test_content = "This is a test file for training. " * 10
        with tempfile.NamedTemporaryFile(
            mode = "w", suffix = ".txt", delete = False
        ) as f:
            f.write(test_content)
            test_file = f.name

        try:
            text_dataset = loader.load_from_file(test_file, return_tokenized = False)
            assert len(text_dataset) > 0
            assert "text" in text_dataset.column_names

            tokenized_dataset = loader.load_from_file(test_file, return_tokenized = True)
            assert len(tokenized_dataset) > 0
            assert "input_ids" in tokenized_dataset.column_names
            assert "attention_mask" in tokenized_dataset.column_names
            assert "labels" in tokenized_dataset.column_names

            first = tokenized_dataset[0]
            assert len(first["input_ids"]) == len(first["attention_mask"])
            assert first["labels"] == first["input_ids"]
        finally:
            os.unlink(test_file)

    def test_text_preprocessor(self):
        from unsloth.dataprep.raw_text import TextPreprocessor

        preprocessor = TextPreprocessor()
        clean = preprocessor.clean_text("  messy   text  \n\n\n  ")
        assert "messy text" in clean

    def test_loader_validation(self):
        from unsloth.dataprep.raw_text import RawTextDataLoader

        class MockTokenizer:
            def __init__(self):
                self.eos_token = "</s>"
                self.eos_token_id = 2

            def __call__(self, text, **kwargs):
                return {"input_ids": list(range(len(text.split())))}

            def decode(self, ids, **kwargs):
                return " ".join(str(i) for i in ids)

        tokenizer = MockTokenizer()

        with pytest.raises(ValueError, match = "chunk_size must be positive"):
            RawTextDataLoader(tokenizer, chunk_size = 0, stride = 2)

        with pytest.raises(ValueError, match = "stride"):
            RawTextDataLoader(tokenizer, chunk_size = 5, stride = 10)


class TestRegistry:
    """Test that model registry works on CPU."""

    def test_register_models(self):
        from unsloth.registry import register_models, MODEL_REGISTRY

        register_models()
        assert len(MODEL_REGISTRY) > 0

    def test_search_models(self):
        from unsloth.registry import search_models

        results = search_models(base_name = "Llama")
        assert len(results) > 0


class TestAutoInstall:
    """Test that auto_install handles CPU gracefully."""

    def test_auto_install_no_crash(self):
        import subprocess

        result = subprocess.run(
            [sys.executable, "unsloth/_auto_install.py"],
            capture_output = True,
            text = True,
            cwd = os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0
        assert "CPU debug mode" in result.stdout
