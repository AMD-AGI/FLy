from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

# Torch types only used for hints; library itself is not required to import here
try:
    import torch
    LongTensor = torch.LongTensor
except Exception:  # pragma: no cover
    LongTensor = Any  # type: ignore

import sys
import os
sys.path.insert(0, "/workspace/Spec-Bench")


# Import the concrete method implementations (these files are in your repo)
from evaluation.inference_lookahead import lookahead_forward
from evaluation.inference_recycling import recycling_forward
# from evaluation.inference_rest import rest_forward
from evaluation.inference_space import space_forward

ReturnType = Tuple[LongTensor, int, int, List[int]]
# -> (output_ids, new_token, idx+1, accept_length_list)


class SpecBenchMethod:
    """
    A tiny adapter that unifies multiple Spec‑Bench speculative decoding methods
    behind a single `.generate()` call.

    Methods supported:
      - "lookahead"  -> inference_lookahead.lookahead_forward
      - "recycling"  -> inference_recycling.recycling_forward
      - "rest"       -> inference_rest.rest_forward
      - "space"      -> inference_space.space_forward

    All backends are expected to return:
        (output_ids, new_token, idx_plus_1, accept_length_list)

    Example
    -------
    >>> sb = SpecBenchMethod(method="lookahead", model=model, tokenizer=tok, max_new_tokens=512)
    >>> out_ids, new_tok, idx_plus_1, acc_list = sb.generate(inputs)
    """

    _REGISTRY = ("lookahead", "recycling", "rest", "space")

    def __init__(
        self,
        method: str,
        model: Any,
        tokenizer: Any,
        max_new_tokens: int = 1024,
        device: Optional[Any] = None,
    ) -> None:
        method = (method or "").lower()
        if method not in self._REGISTRY:
            raise ValueError(f"Unknown method '{method}'. Supported: {list(self._REGISTRY)}.")
        self.method = method
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = int(max_new_tokens)
        self.device = device  # kept for future use; most backends handle device internally

    # ---------------------- public API ----------------------

    def generate(
        self,
        inputs: Any,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **overrides: Any,
    ) -> ReturnType:
        """
        Run the chosen method and return (output_ids, new_token, idx+1, accept_length_list).

        Parameters
        ----------
        inputs : tokenizer output (must have .input_ids; .attention_mask optional)
        max_new_tokens : int, optional
            Overrides the default set at construction.
        temperature : float, optional
        do_sample : bool, optional
        overrides : dict
            Extra keyword arguments forwarded to the specific backend, e.g.:
              - recycling: RECYCLE_STAR=..., RECYCLE_TOKEN=...
              - rest: datastore=..., num_draft=..., token_spans=..., max_steps=...
              - space: MASK_ID=..., MASK_NUM=..., USE_CACHE=..., MAX_NEW_TOKENS=...
        """
        mnt = int(self.max_new_tokens if max_new_tokens is None else max_new_tokens)

        # Default sampling args if the backend accepts them
        if temperature is None:
            temperature = 0.0
        if do_sample is None:
            do_sample = False

        if self.method == "lookahead":
            from model.lade.utils import augment_all, config_lade
            augment_all()
            config_lade(LEVEL=3, WINDOW_SIZE=10, GUESS_SET_SIZE=10, DEBUG=0,
                            USE_FLASH=0, DIST_WORKERS=len(os.environ.get("HIP_VISIBLE_DEVICES").split(",")))
            # lookahead_forward signature (in your file) is:
            #   lookahead_forward(inputs, model, tokenizer, max_new_tokens, [maybe temperature, do_sample])
            try:
                # Preferred: with (temp, do_sample) if supported
                return lookahead_forward(inputs, self.model, self.tokenizer, mnt, temperature, do_sample)
            except TypeError:
                # Fallback: older signature without sampling args
                return lookahead_forward(inputs, self.model, self.tokenizer, mnt)

        elif self.method == "recycling":
            # recycling_forward(
            #   inputs, model, tokenizer, max_new_tokens,
            #   temperature=0.0, do_sample=False,
            #   RECYCLE_STAR=..., RECYCLE_TOKEN=..., ...)
            return recycling_forward(
                inputs,
                self.model,
                self.tokenizer,
                mnt,
                temperature=temperature,
                do_sample=do_sample,
                **overrides,
            )

        elif self.method == "rest":
            # rest_forward(
            #   inputs, model, tokenizer, max_new_tokens,
            #   temperature=0.0, do_sample=0.0?,  # per your file defaults
            #   datastore=None, num_draft=64, token_spans=None, max_steps=512, ...)
            return rest_forward(
                inputs,
                self.model,
                self.tokenizer,
                mnt,
                temperature=temperature,
                do_sample=do_sample,  # will be ignored if the file uses another param name
                **overrides,
            )

        elif self.method == "space":
            # space_forward(
            #   inputs, model, tokenizer, max_new_tokens,
            #   temperature=0.0, do_sample=False,
            #   MASK_ID=32002, MASK_NUM=5, USE_CACHE=True, MAX_NEW_TOKENS=512)
            # NOTE: This backend expects both max_new_tokens (positional) and MAX_NEW_TOKENS (keyword)
            #       We pass MAX_NEW_TOKENS=mnt unless explicitly overridden.
            overrides = {"MAX_NEW_TOKENS": mnt, **overrides}
            return space_forward(
                inputs,
                self.model,
                self.tokenizer,
                mnt,
                temperature=temperature,
                do_sample=do_sample,
                **overrides,
            )

        else:  # pragma: no cover
            raise RuntimeError(f"Unreachable method: {self.method}")
