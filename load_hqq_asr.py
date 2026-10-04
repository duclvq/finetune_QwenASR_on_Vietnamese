"""Load an HQQ-quantized Qwen3-ASR checkpoint (q8/q4 subfolders of the HF repo) with the qwen-asr wrapper.

HQQ keeps scales/zeros in fp32, which clashes with the bf16 audio encoder and the wrapper's dtype detection,
so three small patches are applied after loading. Needs: pip install qwen-asr hqq
"""
import torch
from qwen_asr import Qwen3ASRModel


def load_hqq_asr(path, device="cuda:0"):
    from hqq.core.quantize import HQQLinear
    w = Qwen3ASRModel.from_pretrained(path, dtype=torch.bfloat16, device_map={"": device},
                                      max_inference_batch_size=8, max_new_tokens=300)
    type(w.model).dtype = property(lambda self: torch.bfloat16)   # wrapper casts features to model.dtype
    HQQLinear.matmul = lambda self, x, transpose=True: torch.matmul(  # run the matmul in activation dtype
        x, self.dequantize().to(x.dtype).t() if transpose else self.dequantize().to(x.dtype))
    for m in w.model.modules():
        if type(m).__name__ == "HQQLinear":
            if getattr(m, "bias", None) is not None:
                m.bias.data = m.bias.data.to(torch.bfloat16)
        else:
            for p in m.parameters(recurse=False):
                if p.dtype == torch.float32:
                    p.data = p.data.to(torch.bfloat16)
    return w.eval() if hasattr(w, "eval") else w
