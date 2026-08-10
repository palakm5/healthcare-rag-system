"""
MedGemma-4B LLM Client (text-only path).

Uses google/medgemma-4b-it via Hugging Face Transformers with the model's
native chat template (apply_chat_template via AutoProcessor). This client
is intentionally kept separate from the NVIDIA and Ollama clients — it
follows the same generate() interface so it can be dropped into the
existing Generator class without any changes there.

⚠️  GATING:  google/medgemma-4b-it is a GATED model on Hugging Face.
    Before using this client you must:
      1. Accept the Health AI Developer Foundations terms of use at
         https://huggingface.co/google/medgemma-4b-it
      2. Set the HF_TOKEN environment variable (or .env file) to your
         Hugging Face access token: HF_TOKEN=hf_...

⚠️  HARDWARE:  The 4B model in bfloat16 requires ~8–10 GB VRAM.
    On a smaller GPU or CPU, use quantization="4bit" or quantization="8bit"
    (requires the bitsandbytes package: pip install bitsandbytes>=0.43.0).
    On Apple Silicon (M-series) MPS acceleration is used automatically.

Interface compatibility
-----------------------
The generate() method signature matches NVIDIAClient and OllamaClient
so any code that accepts an llm_client callable will work with this client
unchanged. The extra `image` parameter is reserved for future multimodal
use; passing it now is a no-op on this text-only path.
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lazy imports — heavy torch/transformers imports are deferred to __init__
# so that importing this module does not fail on machines without GPU/HF deps
# until the client is actually instantiated.
_transformers_available = False
try:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    _transformers_available = True
except ImportError:
    pass


class MedGemmaClient:
    """
    Client for google/medgemma-4b-it via Hugging Face Transformers.

    Example (text-only, GPU):
        client = MedGemmaClient()
        answer = client.generate("What are the symptoms of tuberculosis?")

    Example (4-bit quantized, for smaller GPUs / CPU):
        client = MedGemmaClient(quantization="4bit")
        answer = client.generate("Explain ARDS management.")

    Example (future multimodal call — image slot reserved):
        # image parameter accepted but ignored on this text-only path
        answer = client.generate("Describe the findings.", image=None)
    """

    MODEL_ID_DEFAULT = "google/medgemma-4b-it"

    def __init__(
        self,
        model_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        quantization: Optional[str] = None,
        torch_dtype: Optional[str] = "bfloat16",
    ):
        """
        Load the MedGemma model and processor.

        Args:
            model_id: Hugging Face model ID. Defaults to
                      settings.MEDGEMMA_MODEL_ID or "google/medgemma-4b-it".
            hf_token: Hugging Face access token. Falls back to the HF_TOKEN
                      environment variable / settings.HF_TOKEN. Required
                      because the model is gated.
            device: "cuda", "mps", or "cpu". Auto-detected if not set:
                    CUDA → MPS → CPU.
            quantization: None (full precision, default), "4bit", or "8bit".
                          Requires bitsandbytes>=0.43.0 for 4bit/8bit.
            torch_dtype: Model weight dtype. "bfloat16" recommended for GPU.
                         Ignored when quantization is set (bitsandbytes
                         manages dtype internally).
        """
        if not _transformers_available:
            raise ImportError(
                "transformers, torch, and accelerate are required for MedGemmaClient. "
                "Install with: pip install transformers>=4.51.0 torch>=2.2.0 accelerate>=0.30.0"
            )

        # ── Resolve model ID ───────────────────────────────────────────
        if model_id is None:
            try:
                from config.settings import settings
                model_id = settings.MEDGEMMA_MODEL_ID
            except Exception:
                model_id = self.MODEL_ID_DEFAULT
        self.model_id = model_id

        # ── Resolve HF token ───────────────────────────────────────────
        token = hf_token or os.getenv("HF_TOKEN")
        if not token:
            try:
                from config.settings import settings
                token = settings.HF_TOKEN
            except Exception:
                pass
        if not token:
            raise ValueError(
                "HF_TOKEN is required to load gated MedGemma models. "
                "Set it in your environment: export HF_TOKEN=hf_... "
                "and accept the license at https://huggingface.co/google/medgemma-4b-it"
            )
        self._token = token

        # ── Resolve device ─────────────────────────────────────────────
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                logger.warning(
                    "No GPU detected. Running MedGemma on CPU — inference will be "
                    "very slow. Consider using quantization='4bit' and a machine "
                    "with a CUDA or Apple Silicon GPU."
                )
        self.device = device

        # ── Build quantization config ──────────────────────────────────
        quant_config = None
        load_kwargs = {}

        if quantization == "4bit":
            logger.info("Using 4-bit quantization (bitsandbytes).")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = quant_config
        elif quantization == "8bit":
            logger.info("Using 8-bit quantization (bitsandbytes).")
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["quantization_config"] = quant_config
        elif quantization is not None:
            raise ValueError(
                f"Unsupported quantization='{quantization}'. "
                "Use None, '4bit', or '8bit'."
            )

        if quantization is None and torch_dtype:
            dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                         "float32": torch.float32}
            load_kwargs["torch_dtype"] = dtype_map.get(torch_dtype, torch.bfloat16)

        # ── Load processor and model ───────────────────────────────────
        logger.info(
            "Loading MedGemma processor from '%s' (device=%s, quantization=%s)...",
            self.model_id, self.device, quantization,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, token=self._token
        )

        logger.info("Loading MedGemma model weights — this may take a minute...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            token=self._token,
            device_map=self.device if quantization else None,
            **load_kwargs,
        )

        # Move to device when not using device_map (quantized models place
        # themselves automatically)
        if quantization is None:
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("MedGemma model loaded successfully on %s.", self.device)

    # ─────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        image=None,  # Reserved for future multimodal use — currently ignored
    ) -> str:
        """
        Generate a response from MedGemma given a text prompt.

        Uses the model's chat template (apply_chat_template) which produces
        the correct <start_of_turn> / <end_of_turn> formatting required by
        the instruction-tuned variant.

        Args:
            prompt: The prompt text (typically a RAG prompt with context).
            temperature: Sampling temperature. Lower = more deterministic.
            max_new_tokens: Maximum number of new tokens to generate.
            top_p: Nucleus sampling parameter.
            image: Reserved — not used on this text-only path. Pass None.
                   Future image-input support can be added here without
                   restructuring the rest of the interface.

        Returns:
            Generated answer string (prompt echo stripped).
        """
        if image is not None:
            logger.warning(
                "image parameter passed to MedGemmaClient.generate() but "
                "image-input is not yet implemented. The image will be ignored."
            )

        # Build message list using the chat format MedGemma expects.
        # The processor's apply_chat_template handles the Gemma3
        # <start_of_turn> / <end_of_turn> token insertion.
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]

        # Format to a string using the model's jinja chat template.
        # add_generation_prompt=True appends "<start_of_turn>model\n"
        # which signals the model to begin generating its response.
        formatted = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Tokenise and move to device
        inputs = self.processor(
            text=formatted,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                do_sample=temperature > 0,
            )

        # Strip the prompt tokens — decode only the newly generated tokens
        generated_ids = output_ids[0][input_len:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    def close(self):
        """
        Release model resources.

        Moves the model to CPU and clears the CUDA cache if applicable.
        Provided for interface compatibility with NVIDIAClient / OllamaClient.
        """
        try:
            if hasattr(self, "model"):
                self.model.to("cpu")
                if self.device == "cuda":
                    import torch
                    torch.cuda.empty_cache()
        except Exception:
            pass
