import copy
import logging
import os
import json
os.environ["HF_TOKEN"] = "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" 
# os.environ["HF_TOKEN"] = "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"
import sys
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import jinja2
import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from accelerate.utils import get_max_memory
from huggingface_hub import HfApi
from packaging import version
from tqdm import tqdm
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    configure_pad_token,
    get_dtype,
    handle_stop_sequences,
    pad_and_concat,
    postprocess_generated_text,
    stop_sequences_criteria,
)

#jz0728
# from lm_eval.models.revise_decoding import SPDGenerate
# from lm_eval.models.revise_decoding_new import SPDGenerate
# from lm_eval.models.revise_decoding_show_one_example import SPDGenerate
# from lm_eval.models.revise_decoding_walltime import SPDGenerate
from lm_eval.models.FLy import SPDGenerate

# sys.path.insert(0, "/workspace/EAGLE")
# from eagle.model.ea_model import EaModel
import sys
import os
# sys.path.insert(0, "/workspace/Spec-Bench")

# from lm_eval.models.spec_bench_method import SpecBenchMethod

import time
 


if TYPE_CHECKING:
    from transformers.quantizers import AutoQuantizationConfig

eval_logger = logging.getLogger(__name__)

# jz0912
import re
def split_by_seed_marker(pattern, text):
    """
    在 text 中查找 'Here is the completed function:'，
    返回 (seed前的字符串, seed后的字符串)
    如果找不到则第二部分返回空字符串。
    """
    m = re.search(pattern, text)
    if not m:
        eval_logger.warning(f"No {pattern} in the {text}")
        return text, ""
    
    idx = m.start()
    return text[:idx], text[idx:]


@register_model("hf-auto", "hf", "huggingface")
class HFLM(TemplateLM):
    """
    An abstracted Huggingface model class. Enables usage with both models of
    `transformers.AutoModelForCausalLM` and `transformers.AutoModelForSeq2SeqLM` classes.

    Supports data-parallel multi-GPU with HF Accelerate.
    """

    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        backend: Literal["default", "causal", "seq2seq"] = "default",
        # override whether the model should be treated as decoder-only (causal) or encoder-decoder (seq2seq)
        revision: Optional[str] = "main",
        subfolder: str = "",
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        softmax_dtype: Optional[Union[str, torch.dtype]] = None,
        mixed_precision_dtype: Optional[Union[str, torch.dtype]] = None,
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        parallelize: Optional[bool] = False,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        # end token for thinking, either the string or int token id.
        # splits to get response after this token (if provided).
        think_end_token: Union[str, int, None] = None,
        total_gen_tok=1024,
        enable_statistics: Optional[bool] = False,
        config_path=None,
        **kwargs,
    ) -> None:
        super().__init__()
        # optionally: take in an already-initialized transformers.PreTrainedModel
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                json_config = json.load(f)
        
        for key in json_config:
            if key in kwargs:
                json_config[key] = kwargs[key]
                eval_logger.info(f"Parameter '{key}' overridden by kwargs. "
                                 f"JSON value: {json_config[key]} -> kwargs value: {kwargs[key]}")
        self.json_config = json_config

        # 打印最终使用的配置（更新后的配置）
        eval_logger.info("=" * 80)
        eval_logger.info("Final Configuration (after kwargs updates):")
        eval_logger.info("=" * 80)
        eval_logger.info(json.dumps(json_config, indent=2, ensure_ascii=False))
        eval_logger.info("=" * 80)

        # jz0728
        self.target_model_path = json_config["target_model_path"]
        self.use_sd = json_config["use_sd"]
        self.is_qwen = json_config["is_qwen"]
        self.use_eagle2 = json_config["use_eagle2"]
        self.use_eagle3 = json_config["use_eagle3"]
        self.spec_bench_method = json_config["spec_bench_method"]
        self.enable_thinking = json_config["enable_qwen_thinking"]
        self.temperature = json_config["temperature"]
        self.pretrained = pretrained
        

        if not isinstance(pretrained, str):
            eval_logger.warning(
                "`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not launch via accelerate or use `parallelize=True` if passing an existing model this way."
            )
            assert not parallelize, (
                "`parallelize=True` is not compatible with passing pre-initialized model to `pretrained`"
            )
            self._model = pretrained
            self._device = self._model.device
            self._config = self._model.config
            gpus = 0

        else:
            assert isinstance(device, str)
            assert isinstance(pretrained, str)
            assert isinstance(batch_size, (int, str))

            gpus = torch.cuda.device_count()
            accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
            accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
            if accelerator.num_processes > 1:
                self.accelerator = accelerator

            if "npu" in accelerator.device.type:
                gpus = torch.npu.device_count()

            # using one process with no model parallelism
            if not (parallelize or accelerator.num_processes > 1):
                # use user-passed device
                device_list = set(
                    ["cuda", "cpu"]
                    + [f"cuda:{i}" for i in range(gpus)]
                    + ["mps", "mps:0"]
                    + [f"npu:{i}" for i in range(gpus)]
                )
                if device and device in device_list:
                    self._device = torch.device(device)
                    eval_logger.info(f"Using device '{device}'")
                    if device in ("mps", "mps:0") and version.parse(
                        torch.__version__
                    ) < version.parse("2.1"):
                        raise RuntimeError(
                            f"mps requires torch >= 2.1. You have {torch.__version__}"
                        )
                else:
                    eval_logger.info("Device not specified")
                    eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                    self._device = (
                        torch.device("cuda")
                        if torch.cuda.is_available()
                        else torch.device("cpu")
                    )
            else:  # Parallelism managed by accelerate
                if device != "cuda":
                    eval_logger.info(
                        f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                    )
                # TODO: include in warning that `load_in_8bit` etc. affect this too
                self._device = (
                    self.accelerator.device
                    if hasattr(self, "accelerator")
                    else torch.device(device)
                )

            revision = str(revision)  # cast to string if not already one

            self._get_config(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
                gguf_file=gguf_file,
                subfolder=subfolder,
            )

            # determine which of 'causal' and 'seq2seq' backends to use for HF models
        self._get_backend(
            config=self.config, backend=backend, trust_remote_code=trust_remote_code
        )

        # load tokenizer so we know tokenizer vocabulary size before loading model and PEFT
        self._create_tokenizer(
            pretrained,
            tokenizer,
            revision=revision,
            subfolder=subfolder,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
            gguf_file=gguf_file,
            add_bos_token=add_bos_token,
        )

        if (
            quantization_config := getattr(self.config, "quantization_config", None)
        ) is not None and isinstance(quantization_config, dict):
            from transformers.quantizers import AutoQuantizationConfig

            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

        # if we passed `pretrained` as a string, initialize our model now
        if isinstance(pretrained, str):
            self._create_model(
                pretrained=pretrained,
                revision=revision,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                parallelize=parallelize,
                gpus=gpus,
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                peft=peft,
                delta=delta,
                autogptq=autogptq,
                gptqmodel=gptqmodel,
                gguf_file=gguf_file,
                quantization_config=quantization_config,
                subfolder=subfolder,
                **kwargs,
            )

        # access self._model through self.model property outside this method
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()

        self.think_end_token = (
            int(think_end_token)
            if (isinstance(think_end_token, str) and think_end_token.isdigit())
            else think_end_token
        )
        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        # select (or create) a pad token to use
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)

        self.add_bos_token = add_bos_token
        if "gemma" in getattr(self.config, "model_type", ""):
            self.add_bos_token = True
            eval_logger.info(
                f"Model type is '{self.config.model_type}', part of the Gemma family--a BOS token will be used as Gemma underperforms without it."
            )

        self._max_length = max_length
        
        self.delta = delta
        self.peft = peft
        self.revision = revision
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size
        self.softmax_dtype = (
            get_dtype(softmax_dtype) if softmax_dtype is not None else None
        )
        self.mixed_precision_dtype = (
            get_dtype(mixed_precision_dtype)
            if mixed_precision_dtype is not None
            else None
        )

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                # TODO: can remove this whole snippet except in the mps case, perhaps?
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    # place model onto device requested manually,
                    # if not using HF Accelerate or device_map
                    # or any other option that preloads model onto device
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            # multigpu data-parallel support when launched with accelerate
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator

                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )
        
        # jz0728
        if self.use_sd:
            eval_logger.info(f"use_sd is True")
            spd_args = {
                "k":self.json_config["spd_k"],
                "total_gen_tok":total_gen_tok,
                "enable_fly":self.json_config["enable_fly"],
                "win_len":self.json_config["win_len"],
                "entropy_thre":self.json_config["entropy_thre"],
                "use_ngram":self.json_config["use_ngram"],
                "max_ngram_size":self.json_config["max_ngram_size"],
                "num_ngram_pred_tokens":self.json_config["num_ngram_pred_tokens"],
                "verbose":self.json_config["verbose"],
                "abla_no_window":self.json_config["abla_no_window"],
                "enable_statistics":enable_statistics,  # 指标统计开关，从命令行参数传入，默认关闭
            }

            if self.json_config['dual_tok']:
                self.spd_gen = SPDGenerate(self.model, self.target_model, self.tokenizer, eval_logger, spd_args, draft_tokenizer=self.draft_tokenizer)
            else:
                self.spd_gen = SPDGenerate(self.model, self.target_model, self.tokenizer, eval_logger, spd_args)

        # jz0806
        
        elif self.use_eagle3 or self.use_eagle2:
            eval_logger.info(f"use_eagle3 is {self.use_eagle3}, use_eagle2 is {self.use_eagle2}")
            
            self.totaltoken_list = []
            self.totaltime_list = []
            self.round_list = []
            

            if self.use_eagle3:
                use_eagle3 = True
            elif self.use_eagle2:
                use_eagle3 = False


            self.eagle_model = EaModel.from_pretrained(
                                base_model_path=self.target_model_path,
                                ea_model_path=self.json_config["ea_model_path"],
                                total_token=48,
                                depth=6,
                                top_k=10,
                                torch_dtype=torch.bfloat16,
                                low_cpu_mem_usage=True,
                                device_map="auto",
                                use_eagle3=use_eagle3,
                            )
        elif len(self.spec_bench_method) > 0:

            self.totaltoken_list = []
            self.totaltime_list = []
            self.round_list = []

            self.sb = SpecBenchMethod(self.spec_bench_method, self.target_model, self.tokenizer, total_gen_tok, self.target_model.device)





    def _get_accelerate_args(
        self,
        parallelize: Optional[bool] = None,
        device_map: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        gpus: Optional[int] = None,
    ) -> dict:
        """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""
        num_local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        num_machines = int(os.environ.get("WORLD_SIZE", 0)) // num_local_processes
        if (
            num_machines == 0
            and hasattr(self, "accelerator")
            and self.accelerator is not None
        ):
            eval_logger.info(
                "We are not in a distributed setting for accelerate. Setting model_parallel to False."
            )
            parallelize = False

        if parallelize is None:
            # If parallelism is unset by the user, we automatically assign model parallelism
            # if enough extra GPUs are available
            max_memory_all_gpus = get_max_memory()
            # We just want gpu, not cpu, max memory
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            parallelize = bool(num_local_processes < len(max_memory_all_gpus))
            eval_logger.info(
                f"Setting model parallel to {parallelize} since "
                f"the number of local processes is {num_local_processes} "
                f"and the number of GPUs is {len(max_memory_all_gpus)}"
            )

        args = {}
        if parallelize:  # Model parallelism will be used
            max_memory = {}
            if max_memory_per_gpu is not None:  # Using the provided memory requirements
                max_memory_per_gpu_map = {
                    device_idx: max_memory_per_gpu for device_idx in range(gpus)
                }
            else:  # Estimating the possible memory requirements
                max_memory_all_gpus = get_max_memory()
                if "cpu" in max_memory_all_gpus:
                    del max_memory_all_gpus["cpu"]
                if not hasattr(self, "accelerator"):
                    max_memory_per_gpu_map = {
                        k: v for k, v in max_memory_all_gpus.items()
                    }
                else:
                    # use only 1 / num_processes of the GPUs if we are running under accelerate launch
                    max_memory_per_gpu_map = {
                        k: v
                        for k, v in max_memory_all_gpus.items()
                        if k % num_local_processes
                        == (self.accelerator.process_index % num_local_processes)
                    }
            args["max_memory"] = max_memory_per_gpu_map
            args["device_map"] = "auto" if device_map is None else device_map
            eval_logger.info(
                f"Model parallel was set to True, setting max memory per GPU to {max_memory_per_gpu_map} and device map to {args.get('device_map')}"
            )

            if max_cpu_memory is not None:
                max_memory["cpu"] = max_cpu_memory

            args["offload_folder"] = offload_folder
        elif (
            device_map is None
        ):  # No model parallelism, we use the default provided device for our model
            if hasattr(self, "accelerator"):
                device_map = {"": f"{self.accelerator.device}"}
            else:
                device_map = {"": str(self.device)}
            args["max_memory"] = None
            args["device_map"] = device_map
            eval_logger.info(
                f"Model parallel was set to False, max memory was not set, and device map was set to {device_map}"
            )
        else:
            args["max_memory"] = None
            args["device_map"] = None
            eval_logger.info("Model parallel was set to False.")

        return args

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model
        
    @property
    def target_model(self):
        # returns the model, unwrapping it if using Accelerate
        if self.use_sd or len(self.spec_bench_method)>0:
            if hasattr(self, "accelerator"):
                return self.accelerator.unwrap_model(self._target_model)
            else:
                return self._target_model
        else:
            raise ValueError

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _get_backend(
        self,
        config: Union[transformers.PretrainedConfig, transformers.AutoConfig],
        backend: Literal["default", "causal", "seq2seq"] = "default",
        trust_remote_code: Optional[bool] = False,
    ) -> None:
        """
        Helper method during initialization.
        Determines the backend ("causal" (decoder-only) or "seq2seq" (encoder-decoder)) model type to be used.
        sets `self.AUTO_MODEL_CLASS` appropriately if not already set.

        **If not calling HFLM.__init__() or HFLM._get_backend() within a subclass of HFLM,
        user must set `self.backend` to be either "causal" or "seq2seq" manually!**
        """

        assert backend in ["default", "causal", "seq2seq"]

        if backend != "default":
            # if we've settled on non-default backend, use that manually
            if backend == "causal":
                self.backend = backend
            elif backend == "seq2seq":
                self.backend = backend
            eval_logger.info(
                f"Overrode HF model backend type, and using type '{self.backend}'"
            )
        else:
            # determine and use the default HF backend for this model, based on its config + metadata.
            if (
                getattr(config, "model_type")
                in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
            ):
                # first check if model type is listed under seq2seq models, since some
                # models like MBart are listed in both seq2seq and causal mistakenly in HF transformers.
                # these special cases should be treated as seq2seq models.
                self.backend = "seq2seq"
                eval_logger.debug(f"Using model type '{self.backend}'")
            elif (
                getattr(self.config, "model_type") in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            ):
                self.backend = "causal"
                eval_logger.debug(f"Using model type '{self.backend}'")
            else:
                if not trust_remote_code:
                    eval_logger.warning(
                        "HF model type is neither marked as CausalLM or Seq2SeqLM. \
                    This is expected if your model requires `trust_remote_code=True` but may be an error otherwise."
                        "Setting backend to causal"
                    )
                # if model type is neither in HF transformers causal or seq2seq model registries
                # then we default to assuming AutoModelForCausalLM
                self.backend = "causal"
                eval_logger.info(
                    f"Model type cannot be determined. Using default model type '{self.backend}'"
                )

        if self.AUTO_MODEL_CLASS is None:
            if self.backend == "causal":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
            elif self.backend == "seq2seq":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM

    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
        gguf_file: Optional[str] = None,
        subfolder: str = "",
    ) -> None:
        """Return the model config for HuggingFace models"""
        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
            gguf_file=gguf_file,
            subfolder=subfolder,
        )

    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        quantization_config: Optional["AutoQuantizationConfig"] = None,
        subfolder: str = "",
        **kwargs,
    ) -> None:
        """
        Initializes an HF or HF-compatible PreTrainedModel from scratch
        inside HFLM, using the kwargs passed into self.__init__().

        Also handles functionality such as AutoGPTQ usage and PEFT wrapping.

        For future similar extensions to AutoGPTQ that are not core to HF's ecosystem,
        (such as PyTorch models that are nearly, but not quite, fully mirroring
        HF's public interface relied on in this HFLM class)
        please consider subclassing HFLM and overriding this and other methods as needed.
        """

        model_kwargs = kwargs if kwargs else {}

        # Filter out custom configuration parameters that should not be passed to model.from_pretrained
        # These parameters are handled in __init__ and stored in self.json_config
        custom_params = set(self.json_config.keys()) if hasattr(self, 'json_config') else set()
        custom_params.add('config_path')  # Also filter out config_path
        for param in custom_params:
            model_kwargs.pop(param, None)  # Remove if exists, ignore if not

        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=kwargs.get("device_map", None),  # device_map= "cuda:0"  kwargs.get("device_map", None)
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )

        if not autogptq and not gptqmodel:
            if model_kwargs.get("load_in_4bit", None):
                assert transformers.__version__ >= "4.30.0", (
                    "load_in_4bit requires transformers >= 4.30.0"
                )
            if transformers.__version__ >= "4.30.0":
                if model_kwargs.get("load_in_4bit", None):
                    if model_kwargs.get("bnb_4bit_compute_dtype", None):
                        model_kwargs["bnb_4bit_compute_dtype"] = get_dtype(
                            model_kwargs["bnb_4bit_compute_dtype"]
                        )

            # jz0819
            draft_model_kwargs = copy.deepcopy(model_kwargs)
            draft_model_kwargs["device_map"] = {"":0}
            self._model = self.AUTO_MODEL_CLASS.from_pretrained(
                pretrained,
                revision=revision,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                **draft_model_kwargs,
            )
            # jz0728
            # print(f"{revision=}")
            # print(f"{model_kwargs=}")
            # print(f"{subfolder=}")
            if self.use_sd or len(self.spec_bench_method)>0:

                if self.spec_bench_method == "lookahead":
                    from model.lade.utils import augment_all, config_lade
                    augment_all()
                    config_lade(LEVEL=3, WINDOW_SIZE=10, GUESS_SET_SIZE=10, DEBUG=0,
                                    USE_FLASH=0, DIST_WORKERS=len(os.environ.get("HIP_VISIBLE_DEVICES").split(",")))

                self._target_model = self.AUTO_MODEL_CLASS.from_pretrained(
                    self.target_model_path,
                    revision=revision,
                    attn_implementation="sdpa",
                    torch_dtype=get_dtype(dtype),
                    trust_remote_code=trust_remote_code,
                    **model_kwargs,
                )
        else:
            if autogptq and gptqmodel:
                raise ValueError(
                    "Cannot use both 'autogptq' and 'gptqmodel' options at the same time."
                )

            if autogptq:
                try:
                    from auto_gptq import AutoGPTQForCausalLM
                except ModuleNotFoundError as exception:
                    raise type(exception)(
                        "Tried to load auto_gptq, but auto-gptq is not installed ",
                        "please install auto-gptq via pip install lm-eval[gptq] or pip install -e .[gptq]",
                    )

                self._model = AutoGPTQForCausalLM.from_quantized(
                    pretrained,
                    trust_remote_code=trust_remote_code,
                    model_basename=None if autogptq is True else Path(autogptq).stem,
                    use_safetensors=True
                    if autogptq is True
                    else autogptq.endswith(".safetensors"),
                    **model_kwargs,
                )

            if gptqmodel:
                try:
                    from gptqmodel import GPTQModel
                except ModuleNotFoundError as exception:
                    raise type(exception)(
                        "Tried to load gptqmodel, but gptqmodel is not installed ",
                        "please install gptqmodel via `pip install gptqmodel --no-build-isolation` or `pip install lm-eval[gptqmodel] --no-build-isolation`",
                    )

                self._model = GPTQModel.from_quantized(
                    pretrained, trust_remote_code=trust_remote_code, **model_kwargs
                )

        if peft and delta:
            raise ValueError(
                "Cannot use both 'peft' and 'delta' options at the same time."
            )

        if peft:
            from peft import PeftModel
            from peft import __version__ as PEFT_VERSION

            if model_kwargs.get("load_in_4bit", None):
                if version.parse(PEFT_VERSION) < version.parse("0.4.0"):
                    raise AssertionError("load_in_4bit requires peft >= 0.4.0")
            if self._model.config.vocab_size != len(self.tokenizer):
                # resize model for LoRAs with added tokens
                eval_logger.info(
                    f"Model config indicates vocab_size='{self._model.config.vocab_size}', but found tokenizer with vocab size '{len(self.tokenizer)}'. Resizing model embedding layer..."
                )
                self._model.resize_token_embeddings(len(self.tokenizer))
            self._model = PeftModel.from_pretrained(
                self._model, peft, revision=revision
            )
        elif delta:
            if autogptq:
                eval_logger.warning(
                    "Delta weights might trigger unexpected behavior when used with AutoGPTQ."
                )
            _model_delta = self.AUTO_MODEL_CLASS.from_pretrained(
                delta,
                revision=revision,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
            for name, param in self._model.state_dict().items():
                try:
                    param.data += _model_delta.state_dict()[name]
                except KeyError:
                    raise KeyError(f"Delta model is missing weights for layer: {name}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to add delta weights to layer {name}. Error: {e}"
                    )

            del _model_delta

        return None

    def _create_tokenizer(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ],
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        gguf_file: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
        subfolder: Optional[str] = "",
    ) -> None:
        """
        Helper method during initialization.

        Create a tokenizer object corresponding to the correct
        tokenizer for value of `pretrained`, or use the pre-initialized tokenizer passed.
        """
        kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }

        # gguf format embeds tokenizer and is not compatible with hf tokenizer `use_fast` param
        if not tokenizer and gguf_file is not None:
            kwargs["gguf_file"] = gguf_file
        else:
            kwargs["use_fast"] = use_fast_tokenizer

        if add_bos_token:
            kwargs["add_bos_token"] = True

        if subfolder:
            kwargs["subfolder"] = subfolder

        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer, **kwargs
                )
            else:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
        elif self.use_sd:
            model_name = self.target_model_path
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, **kwargs
            )
            if self.json_config['dual_tok']:
                draft_name = self.pretrained
                self.draft_tokenizer = transformers.AutoTokenizer.from_pretrained(
                    draft_name, **kwargs
                )
        else:
            # Get tokenizer based on 'pretrained'
            if isinstance(pretrained, str):
                model_name = pretrained
            else:
                # get the HF hub name via accessor on model
                model_name = self.model.name_or_path
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, **kwargs
            )
        return None

    def _detect_batch_size(self, requests=None, pos: int = 0):
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_length = len(
                (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
            )
            max_context_enc = len(context_enc[-(self.max_length + 1) :])
            max_cont_enc = len(continuation_enc[-(self.max_length + 1) :])
        else:
            max_length = self.max_length
            max_context_enc = max_length
            max_cont_enc = max_length

        # if OOM, then halves batch_size and tries again
        @find_executable_batch_size(starting_batch_size=self.max_batch_size)
        def forward_batch(batch_size):
            if self.backend == "seq2seq":
                length = max(max_context_enc, max_cont_enc)
                batched_conts = torch.ones(
                    (batch_size, length), device=self.device
                ).long()
                test_batch = torch.ones((batch_size, length), device=self.device).long()
                call_kwargs = {
                    "attn_mask": test_batch,
                    "labels": batched_conts,
                }
            else:
                call_kwargs = {}
                test_batch = torch.ones(
                    (batch_size, max_length), device=self.device
                ).long()
            for _ in range(5):
                out = F.log_softmax(  # noqa: F841
                    self._model_call(test_batch, **call_kwargs),
                    dim=-1,
                    dtype=self.softmax_dtype,
                )

            return batch_size

        try:
            batch_size = forward_batch()
        except RuntimeError as e:
            if "No executable batch size found" in str(e):
                batch_size = 1
            else:
                raise

        if self.world_size > 1:
            # if multi-GPU, always take minimum over all selected batch sizes
            max_rnk_bs = torch.tensor([batch_size], device=self.device)
            gathered = (
                self.accelerator.gather(max_rnk_bs).cpu().detach().numpy().tolist()
            )
            batch_size = min(gathered)
            clear_torch_cache()
            return batch_size

        clear_torch_cache()
        return batch_size

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            if self.backend == "causal":
                special_tokens_kwargs = {
                    "add_special_tokens": False or self.add_bos_token
                }
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {}
        if self.backend == "causal":
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len:
            original_lengths = encoding["input_ids"].size(1)
            if original_lengths > left_truncate_len:
                eval_logger.warn(
                    f"Left truncation applied. Original sequence length was {original_lengths}, "
                    f"truncating to last {left_truncate_len} tokens. Some content will be lost.",
                )
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"], encoding

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision_dtype,
                enabled=self.mixed_precision_dtype is not None,
            ):
                if attn_mask is not None or labels is not None:
                    assert attn_mask is not None and labels is not None
                    assert self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM
                    return self.model(
                        input_ids=inps, attention_mask=attn_mask, labels=labels
                    ).logits
                else:
                    assert self.AUTO_MODEL_CLASS in (
                        transformers.AutoModelForCausalLM,
                        transformers.AutoModelForVision2Seq,
                    )
                    return self.model(inps).logits

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            return self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                **generation_kwargs,
            )

    # jz0728
    def _model_generate_sd(self, context, max_length, stop, attention_mask=None, input_context_for_dualtok=None, **generation_kwargs):
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            accepted_token = self.spd_gen.generate_chunks(context, self.temperature)
        _ = self.spd_gen.show_status()
        return accepted_token

    def _model_generate_sb(self, context, max_length, stop, **generation_kwargs):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            out_ids, new_tok, idx_plus_1, acc_list = self.sb.generate(context)

        torch.cuda.synchronize()
        total_time = time.time() - start_time
        self.collect_statistics(total_time,new_tok,idx_plus_1)

        return out_ids


    # jz0806
    def _model_generate_eagle(self, context, max_length, stop, **generation_kwargs):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.inference_mode():
            output_ids, new_token, idx = self.eagle_model.eagenerate(
                    torch.as_tensor(context).cuda(),
                    temperature=0,
                    log=True,
                    is_llama3=True,
                    max_length=10240,
                )

        torch.cuda.synchronize()
        total_time = time.time() - start_time
        self.collect_statistics(total_time,new_token,idx)

        return output_ids


    def collect_statistics(self, total_time, new_token, idx):
        self.totaltoken_list.append(int(new_token))
        self.totaltime_list.append(total_time)
        self.round_list.append(idx)

        mat = sum(self.totaltoken_list) / sum(self.round_list)
        speed = sum(self.totaltoken_list) / sum(self.totaltime_list)

        eval_logger.info(f"Speed:{speed:.2f}|MAT:{mat:.2f}")



    def _select_cont_toks(
        self, logits: torch.Tensor, contlen: int = None, inplen: int = None
    ) -> torch.Tensor:
        if self.backend == "causal":
            assert contlen and inplen, (
                "Must pass input len and cont. len to select scored logits for causal LM"
            )
            # discard right-padding.
            # also discard the input/context tokens. we'll only score continuations.
            logits = logits[inplen - contlen : inplen]
        elif self.backend == "seq2seq":
            assert contlen and not inplen, (
                "Selecting scored logits for Seq2SeqLM requires only cont. len"
            )
            # only discard right-padding.
            # the logits input to this fn only contain decoder-side tokens.
            logits = logits[:contlen]

        return logits

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        # First, collect all windows from all requests
        all_windows = []  # List of (request_idx, window) tuples
        request_window_counts = []  # Track number of windows per request

        for req_idx, (string,) in enumerate(
            tqdm(
                [req.args for req in requests],
                disable=(disable_tqdm or (self.rank != 0)),
            )
        ):
            rolling_token_windows: List[Tuple[List[int], List[int]]] = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            windows = [(None,) + x for x in rolling_token_windows]

            # Store windows with their request index
            all_windows.extend((req_idx, window) for window in windows)
            request_window_counts.append(len(windows))

        # Handle distributed case padding
        pad_amnt = 0
        if self.world_size > 1:
            mytensor = torch.tensor(len(all_windows), device=self.device)
            gathered = self.accelerator.gather(mytensor).cpu().detach().numpy().tolist()
            pad_amnt = max(gathered) - gathered[self.rank]
            if pad_amnt > 0:
                all_windows += pad_amnt * [all_windows[0]]

        all_nlls = []
        batch_size = adaptive_batch_size or self.batch_size
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]
            # Extract just the windows for processing, keeping track of request indices
            batch_indices, batch_windows = zip(*batch)

            batch_nlls = self._loglikelihood_tokens(
                requests=batch_windows,
                disable_tqdm=False,
                override_bs=len(batch_windows),
            )
            # Store results with their request indices
            all_nlls.extend(zip(batch_indices, batch_nlls))

        # Remove padding if necessary
        if (self.world_size > 1) and (pad_amnt > 0):
            all_nlls = all_nlls[:-pad_amnt]

        # Reconstruct per-request loglikelihoods
        loglikelihoods = []
        current_idx = 0
        for window_count in request_window_counts:
            # Get all nlls for this request
            request_nlls = all_nlls[current_idx : current_idx + window_count]
            # Sum up the nlls for this request (discarding is_greedy)
            request_total = sum(nll[0] for _, nll in request_nlls)
            loglikelihoods.append(request_total)
            current_idx += window_count

            string = requests[len(loglikelihoods) - 1].args[0]
            self.cache_hook.add_partial(
                "loglikelihood_rolling", (string,), request_total
            )

        return loglikelihoods

    def _batch_scheduler(self, pos, n_reordered_requests):
        sched = pos // int(len(n_reordered_requests) / self.batch_schedule)
        if sched in self.batch_sizes:
            return self.batch_sizes[sched]
        if (len(self.batch_sizes) > 1) and (
            self.batch_sizes[sched - 1] == self.max_batch_size
        ):
            # if previous batch size is already maximal, skip recomputation
            self.batch_sizes[sched] = self.max_batch_size
            return self.batch_sizes[sched]
        print(
            f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size"
        )
        self.batch_sizes[sched] = self._detect_batch_size(n_reordered_requests, pos)
        print(f"Determined largest batch size: {self.batch_sizes[sched]}")
        return self.batch_sizes[sched]

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts"
            if self.backend == "causal" and self.logits_cache
            else None,
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                if self.backend == "causal":
                    total_length = len(context_enc) + len(continuation_enc)
                    if total_length > self.max_length + 1:
                        eval_logger.warning(
                            f"Combined length of context ({len(context_enc)}) and continuation ({len(continuation_enc)}) "
                            f"exceeds model's maximum length ({self.max_length}). "
                            f"Truncating {total_length - self.max_length + 1} tokens from the left."
                        )
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape
                elif self.backend == "seq2seq":
                    inp = torch.tensor(
                        (context_enc)[-self.max_length :],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape

                    # build encoder attn masks
                    encoder_attns.append(torch.ones_like(inp))

                    cont = torch.tensor(
                        (continuation_enc)[-self.max_length :],
                        # TODO: left-shift these?
                        # TODO: our code assumes we never end up truncating conts for either model type
                        dtype=torch.long,
                        device=self.device,
                    )
                    (contlen,) = cont.shape

                    conts.append(cont)

                    padding_len_cont = (
                        max(padding_len_cont, contlen)
                        if padding_len_cont is not None
                        else contlen
                    )

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            if self.backend == "causal":
                batched_inps = pad_and_concat(
                    padding_len_inp, inps, padding_side="right"
                )  # [batch, padding_len_inp]
            elif self.backend == "seq2seq":
                # TODO: left-pad encoder inps and mask?
                batched_inps = pad_and_concat(
                    padding_len_inp, inps
                )  # [batch, padding_len_inp]
                batched_conts = pad_and_concat(
                    padding_len_cont, conts
                )  # [batch, padding_len_cont]
                batched_encoder_mask = pad_and_concat(
                    padding_len_inp, encoder_attns
                )  # [batch, padding_len_inp]
                call_kwargs = {
                    "attn_mask": batched_encoder_mask,
                    "labels": batched_conts,
                }

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs),
                dim=-1,
                dtype=self.softmax_dtype,
            )  # [batch, padding_length (inp or cont), vocab]

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                    if self.backend == "causal"
                    else None
                )
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(0)  # [1, seq]
                    # Use trailing slice [-cont_toks.shape[1]:] to handle variable length cont_len (but same ctx+cont[:-1]).
                    # i.e. continuations can be sliced at diff points. Collator ensures we have sufficient greedy_tokens
                    # by choosing key with longest cont if group_by="contexts".
                    max_equal = (
                        greedy_tokens[:, -cont_toks.shape[1] :] == cont_toks
                    ).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    if request_str is not None:
                        # special case: loglikelihood_rolling produces a number of loglikelihood requests
                        # all with cache key None. instead do add_partial on the per-example level
                        # in the loglikelihood_rolling() function for those.
                        self.cache_hook.add_partial(
                            "loglikelihood", request_str, answer
                        )
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        def _collate(req: Tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size
            if adaptive_batch_size is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and not adaptive_batch_size
            else None
        )

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                # add EOS token to stop sequences
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            if self.backend == "causal":
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
                assert max_ctx_len > 0, (
                    f"Invalid configuration: requested max tokens to generate ({max_gen_toks}) must be less than model's maximum sequence length ({self.max_length})."
                )
            elif self.backend == "seq2seq":
                # max len for inputs = encoder's whole max_length
                max_ctx_len = self.max_length

            # encode, pad, and truncate contexts for this batch
            # print(f"Before >>> {contexts=}")  # jz0912
            
            if self.json_config['dual_tok']:
                # currently only for HumanEval
                pattern = "Here is the completed function"
                user_cont, assis_cont = split_by_seed_marker(pattern, contexts[0])
                msgs = [{"role": "user", "content": user_cont}]
                
                contexts = self.draft_tokenizer.apply_chat_template(
                    msgs, 
                    add_generation_prompt=True, 
                    tokenize=False
                )
                contexts = (contexts + assis_cont.strip(),)

                if "DeepSeek" in self.target_model_path:
                    contexts_new = contexts + assis_cont
                    contexts_new = contexts_new.replace("<think>", "<think>\n\n</think>\n\nFinal answer:")
                    contexts = (contexts_new,)
                else:
                    contexts = (contexts + assis_cont,)

            # print(f"After >>> {contexts=}")  # jz0912
            # print(f"{self.draft_tokenizer.eos_token}")
            # print(f"{self.target_tokenizer.eos_token}")


            context_enc, attn_masks, original_input_ids = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            context_enc = context_enc.to(self.device)
            attn_masks = attn_masks.to(self.device)
            original_input_ids = original_input_ids.to(self.device)

            # jz0912
            # test_ids = self.tokenizer(contexts, add_special_tokens=False, return_tensors="pt")
            # print(f"{test_ids['input_ids']=}")
            # print(f"{context_enc=}")
            # raise ValueError

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            # jz0728
            if self.use_sd:
                if self.json_config['dual_tok']:
                    input_context_for_dualtok = (user_cont, assis_cont)
                else:
                    input_context_for_dualtok = None

                cont = self._model_generate_sd(
                    context=context_enc,
                    attention_mask=attn_masks,
                    stop=until,
                    input_context_for_dualtok=input_context_for_dualtok,
                    **kwargs,
                )
            elif self.use_eagle3 or self.use_eagle2:
                cont = self._model_generate_eagle(
                    context=context_enc,
                    attention_mask=attn_masks,
                    stop=until,
                    **kwargs,
                ) 
            elif len(self.spec_bench_method) > 0:
                cont = self._model_generate_sb(
                    context=original_input_ids,
                    attention_mask=attn_masks,
                    stop=until,
                    **kwargs,
                )


            else:
                # perform batched generation
                cont = self._model_generate(
                    context=context_enc,
                    attention_mask=attn_masks,
                    stop=until,
                    **kwargs,
                )

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                # discard context + left-padding toks if using causal decoder-only LM
                if self.backend == "causal":
                    cont_toks = cont_toks[context_enc.shape[1] :]

                # Handle integer think_end_token: find last occurrence and strip tokens after it
                if isinstance(self.think_end_token, int):
                    think_token_indices = [
                        i
                        for i, token in enumerate(cont_toks)
                        if token == self.think_end_token
                    ]
                    if think_token_indices:
                        cont_toks = cont_toks[think_token_indices[-1] + 1 :]

                s = self.tok_decode(cont_toks)

                # Strip leading whitespace if we removed thinking tokens
                if isinstance(self.think_end_token, int):
                    s = s.lstrip()

                # Apply post-processing: remove stop sequences and string-based thinking tokens
                s = postprocess_generated_text(
                    generation=s,
                    stop=until,
                    think_end_token=self.think_end_token
                    if isinstance(self.think_end_token, str)
                    else None,
                )
                res.append(s)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        
        # 如果启用了统计，在数据集评估完成后打印全局统计
        if self.use_sd and hasattr(self, 'spd_gen') and self.spd_gen.enable_statistics:
            self.spd_gen.print_global_statistics()

        return res

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        try:
            # if self.enable_thinking:
            #     chat_templated = self.tokenizer.apply_chat_template(
            #         chat_history,
            #         tokenize=False,
            #         add_generation_prompt=add_generation_prompt,
            #         continue_final_message=not add_generation_prompt,
            #     )
            # else:
            #     chat_templated = self.tokenizer.apply_chat_template(
            #         chat_history,
            #         tokenize=False,
            #         add_generation_prompt=add_generation_prompt,
            #         continue_final_message=not add_generation_prompt,
            #         enable_thinking=False
            #     )

            chat_templated = self.tokenizer.apply_chat_template(
                    chat_history,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=not add_generation_prompt,
                )



        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. removing the system role in chat history."
            )
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )

        return chat_templated

    def get_model_info(self) -> dict:
        """
        Method to get Hugging Face model information for experiment reproducibility.
        """

        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            if hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            else:
                return -1

        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            else:
                return ""

        def get_model_sha(pretrained: str, revision: str) -> str:
            try:
                model_info = HfApi().model_info(repo_id=pretrained, revision=revision)
                return model_info.sha
            except Exception as e:
                eval_logger.debug(
                    f"Failed to get model SHA for {pretrained} at revision {revision}. Error: {e}"
                )
                return ""

        model_info = {
            "model_num_parameters": get_model_num_params(self._model),
            "model_dtype": get_model_dtype(self._model),
            "model_revision": self.revision,
            "model_sha": get_model_sha(self.pretrained, self.revision),
        }
        if self.peft:
            model_info["peft_sha"] = get_model_sha(self.peft, self.revision)
        if self.delta:
            model_info["delta_sha"] = get_model_sha(self.delta, self.revision)
        return model_info
