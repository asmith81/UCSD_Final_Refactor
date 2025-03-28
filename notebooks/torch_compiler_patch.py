
import torch
import sys

# Add torch.compiler attribute if missing (needed for newer transformers)
if not hasattr(torch, 'compiler'):
    print("✅ Adding missing torch.compiler attribute")
    class DummyCompiler:
        @staticmethod
        def disable(*args, **kwargs):
            return lambda x: x
    torch.compiler = DummyCompiler()
else:
    print("✅ torch.compiler attribute already exists")

# Verify Transformers can be imported with this patch
try:
    import transformers
    from transformers import LlavaForConditionalGeneration
    print(f"✅ Successfully imported transformers {transformers.__version__} with LlavaForConditionalGeneration")
except Exception as e:
    print(f"❌ Error importing transformers: {e}")
    sys.exit(1)
