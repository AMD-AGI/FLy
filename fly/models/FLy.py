# Copyright © 2026 Advanced Micro Devices, Inc. All rights reserved

import torch
import time
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from functools import cached_property
import os



# torch.multinomial forces a GPU<->CPU sync.
# Therefore, we use an optimized implementation instead that skips the sync.
# Note that we always sample with replacement.
# probs will be modified in place, but this is fine, as we pass
# in a copy already.
# @torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
# def _multinomial(
#     probs: torch.Tensor, # [batch_size, k, vocab_size]
#     num_samples: int,
# ) -> torch.Tensor:

#     q = torch.empty_like(probs)
#     q.exponential_(1.0)
#     return probs.div_(q).argmax(dim=-1).view(-1, num_samples)

def _multinomial(
    probs: torch.Tensor,  # [batch_size, k, vocab_size]
    num_samples: int,
) -> torch.Tensor:
    # 扩展概率张量以支持多次采样
    expanded_probs = probs.unsqueeze(1).expand(
        -1, num_samples, -1, -1
    )  # [batch_size, num_samples, k, vocab_size]
    
    # 生成随机噪声用于 Gumbel-Max 采样
    q = torch.empty_like(expanded_probs)  # [batch_size, num_samples, k, vocab_size]
    q.exponential_(1.0)
    
    # 执行 Gumbel-Max 采样
    samples = (expanded_probs / q).argmax(dim=-1)  # [batch_size, num_samples, k]
    
    return samples  # [batch_size, num_samples, k]

def sample_with_temperature(logits, temp=0, sample_times=1):
    """
    logits: [bs, k, vocab_size]
    return:  [bs, k]
    """

    b, k, v = logits.shape
    if temp > 0:
        logits = logits / temp
        probs = F.softmax(logits, dim=-1)
        next_token = _multinomial(probs, num_samples=sample_times)
    else:
        next_token = logits.argmax(dim=-1)

    return next_token.view(sample_times, k)



class SPDGenerate:
    def __init__(self, draft_model, target_model, tokenizer, cuslog, spd_args, draft_tokenizer=None):
        self.draft_model = draft_model.eval()
        self.target_model = target_model.eval()
        self.tokenizer = tokenizer
        self.cuslog = cuslog
        self.draft_tokenizer = draft_tokenizer
        self.k = spd_args.get("k")
        self.total_gen_tok = spd_args.get("total_gen_tok")

        self.probs_dtype = torch.float32
        self.token_id_dtype = torch.long
        self._num_bonus_tokens = 1

        # self.num_accepted_tokens = torch.tensor(0, dtype=torch.long, device=device)
        # self.num_emitted_tokens  = torch.tensor(0, dtype=torch.long, device=device)
        # self.num_draft_tokens    = 0
        self._counter_inited=False

        # self.cuslog.info(f"{self.draft_model.device=}")
        self.cuslog.info(f"{self.draft_model.device=}, {self.target_model.device=}, {self.target_model.dtype=}")
        self.speed_list, self.mat_list = [], []

        self.enable_fly = spd_args.get("enable_fly")
        self.win_len = spd_args.get("win_len")

        # self.init_entropy_statistics()
        entropy_thre = float(spd_args.get("entropy_thre", 0))
        self.entropy_thre = entropy_thre if entropy_thre > 1e-2 else None

        self.use_ngram       = spd_args.get("use_ngram", False)
        self.max_ngram_size  = spd_args.get("max_ngram_size", 3)
        self.num_ngram_pred_tokens = spd_args.get("num_ngram_pred_tokens", 10)

        self.debug_ngram_accept_num = []
        self.bonus_tok_from_target = 1

        self.verbose = spd_args.get("verbose", False)

        self.abla_no_window = spd_args.get("abla_no_window", False)

        # 指标统计控制开关
        self.enable_statistics = spd_args.get("enable_statistics", False)
        
        # Tree verify 相关参数
        self.tree_verify = spd_args.get("tree_verify", False)
        self.branch_n = spd_args.get("branch_n", 10)  # 每个节点拓展的 top-n
        self.max_nodes_per_level = spd_args.get("max_nodes_per_level", 10)  # 每层最多保留的节点数
        self.max_nodes_global = spd_args.get("max_nodes_global", 100)  # 全局最多保留的节点数
        
        # 当启用 tree_verify 时，禁用 ngram（不兼容）
        if self.tree_verify:
            self.use_ngram = False
        
        # 初始化每个样本的统计变量（会在 reset_counter 中重置）
        self.total_initial_mismatch = 0  # 初始的 mismatch 总数（accepted 为 False 的数量）
        self.total_fly_accepted = 0      # 通过 FLy 算法从 False 变成 True 的数量
        
        # 初始化全局统计变量（整个数据集的累计，不会被 reset_counter 重置）
        self.global_total_initial_mismatch = 0  # 整个数据集的初始 mismatch 总数
        self.global_total_fly_accepted = 0      # 整个数据集的 FLy 接受总数
        self.global_sample_count = 0            # 处理的样本总数



    def _ensure_counters(self, device: torch.device):
        if self._counter_inited:
            if self.num_accepted_tokens.device != device:
                self.num_accepted_tokens = self.num_accepted_tokens.to(device, non_blocking=True)
                self.num_emitted_tokens  = self.num_emitted_tokens.to(device, non_blocking=True)
                self.num_draft_round    = self.num_draft_round.to(device, non_blocking=True)
            return

        # 第一次使用：在当前 device 上创建 0 维 long 标量
        self.num_accepted_tokens = torch.zeros((), dtype=torch.long, device=device)
        self.num_emitted_tokens  = torch.zeros((), dtype=torch.long, device=device)
        self.num_draft_round    = torch.zeros((), dtype=torch.long, device=device)
        self._counter_inited = True


    def _batch_verification(
        self,
        target_logits: torch.Tensor,   # [B, k, V]  注意：传 logits 而非 probs
        draft_token_ids: torch.Tensor, # [B, k]
        temp=0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        批量验证草稿 tokens
        
        输入形状:
            target_logits: [B, k, V] - target 模型的 logits
            draft_token_ids: [B, k] - 草稿模型生成的 token IDs
        
        输出形状:
            accepted: [B, k] - bool 张量，表示每个位置是否接受
            recovered_token_ids: [B, k] - 恢复的 token IDs
        """

        # target 在每个草稿位置的采样或 argmax
        if temp == 0:
            target_ids = sample_with_temperature(target_logits, temp)  # [1, k] (sample_times=1)
        else:
            target_ids = sample_with_temperature(target_logits, temp, 2)  # [2, k] (sample_times=2)

        # 接受：draft 等于 target 的采样结果
        accepted = (target_ids == draft_token_ids)  # [sample_times, k], bool
        accepted = accepted.any(dim=0).unsqueeze(0)  # [1, k], bool - 任一采样匹配即接受

        # 恢复 token：使用第一次采样的结果
        recovered_token_ids = target_ids[0:1,:]  # [1, k]

        return accepted, recovered_token_ids  # ([1, k], [1, k])


    def _batch_modified_rejection_sampling(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        seeded_seqs: Optional[dict[int, torch.Generator]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform modified rejection sampling on each sequence.

        Returns:
            A tuple of two tensors:
            0: A bool tensor of which tokens in each sequence is accepted.
                shape = [batch_size, k]
            1: Token ids sampled from a recovered distribution, to be used
                when a token is rejected.
                shape = [batch_size, k]
        """

        batch_size, k, vocab_size = draft_probs.shape

        # shape [batch_size, k]
        accepted = self._get_accepted(target_probs, draft_probs,
                                      draft_token_ids, seeded_seqs)

        recovered_probs = self._get_recovered_probs(
            target_probs, draft_probs).reshape(batch_size * k, vocab_size)

        # NOTE: the recovered_probs are overwritten by this method.
        # Reshape recovered_probs to [batch_size, k, vocab_size] for _multinomial
        recovered_probs_reshaped = recovered_probs.reshape(batch_size, k, vocab_size)
        recovered_token_ids = _multinomial(
            recovered_probs_reshaped,
            num_samples=1,
        ).reshape(batch_size, k)

        return accepted, recovered_token_ids

    def _get_accepted(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        seeded_seqs: Optional[dict[int, torch.Generator]],
    ) -> torch.Tensor:
        r"""Create bool matrix over the proposed draft tokens. If
        True, then a token can be accepted, else it should be
        rejected.

        Given $q(\hat{x}_{n+1}|x_1, \dots, x_n)$, the probability of
        $\hat{x}_{n+1}$ given context $x_1, \dots, x_n$ according
        to the target model, and $p(\hat{x}_{n+1}|x_1, \dots, x_n)$, the
        same conditional probability according to the draft model, the token
        is accepted with probability:

        $$
        \min\left(1, \frac{q(\hat{x}_{n+1}|x_1, \dots, x_n)}
                        {p(\hat{x}_{n+1}|x_1, \dots, x_n)}\right)
        $$

        This implementation does not apply causality. When using the output,
        if a token is rejected, subsequent tokens should not be used.

        Returns a bool tensor of shape [batch_size, k] specifying which tokens
        are accepted.
        """
        batch_size, k, _ = draft_probs.shape
        
        # bs=1
        # self.cuslog.info(f"{draft_probs.shape=}")
        # self.cuslog.info(f"{draft_token_ids.shape=}")
        selected_draft_probs = torch.gather(draft_probs, dim=-1, index=draft_token_ids.unsqueeze(-1)).squeeze(-1)
        # self.cuslog.info(f"{selected_draft_probs.shape=}")

        # shape [batch_size, k]
        selected_target_probs = torch.gather(target_probs, dim=-1, index=draft_token_ids.unsqueeze(-1)).squeeze(-1)

        
        uniform_rand = torch.rand(batch_size, k, device=target_probs.device)

        capped_ratio = torch.minimum(
            selected_target_probs / selected_draft_probs,
            torch.full((1, ), 1, device=target_probs.device))
        accepted = uniform_rand < capped_ratio

        return accepted

    def _get_recovered_probs(
            self,
            target_probs: torch.Tensor,  # [k, vocab_size]
            draft_probs: torch.Tensor,  # [k, vocab_size]
    ) -> torch.Tensor:
        r"""Create a probability distribution for each proposed token which can
        be sampled if the proposed token is rejected.

        When this routine is applied sequentially, the true distribution of the
        target model is recovered (within hardware numerics).

        The probability distribution used in this rejection case is constructed
        as follows. Given $q(x|x_1, \dots, x_n)$, the probability of
        $x$ given context $x_1, \dots, x_n$ according to the target
        model and $p(x|x_1, \dots, x_n)$, the same conditional probability
        according to the draft model:

        $$
        x_{n+1} \sim (q(x|x_1, \dots, x_n) - p(x|x_1, \dots, x_n))_+
        $$

        where $(f(x))_+$ is defined as:

        $$
        (f(x))_+ = \frac{\max(0, f(x))}{\sum_x \max(0, f(x))}
        $$

        See https://github.com/vllm-project/vllm/pull/2336 for a visualization
        of the draft, target, and recovered probability distributions.

        Returns a tensor of shape [batch_size, k, vocab_size].

        Note: 
            This batches operations on GPU and thus constructs the recovered
            distribution for all tokens, even if they are accepted. This causes
            division-by-zero errors, so we use self._smallest_positive_value to
            avoid that. This introduces some drift to the distribution.
        """
        _, k, _ = draft_probs.shape

        # shape [batch_size, k, vocab_size]
        difference = target_probs - draft_probs

        # TODO(cade): Can we use logprobs instead of probs, and avoid the
        # division-by-zero errors without introducing distribution drift?

        # shape [batch_size, k, vocab_size]
        f = torch.clamp(difference, min=self._smallest_positive_value)

        # shape [batch_size, k, vocab_size]
        recovered_probs = f / torch.sum(f, dim=-1).reshape(-1, k, 1)

        return recovered_probs

    @cached_property
    def _smallest_positive_value(self) -> float:
        """Return the smallest positive value representable by the probs dtype.
        This value is used when constructing a distribution from which to sample
        recovered tokens in the first rejection case.

        See _get_recovered_probs for more details

        Note that this isn't actually the smallest positive value representable
        by float32, but the smallest positive normal value.
        See https://en.wikipedia.org/wiki/Subnormal_number for more information.
        """
        return torch.finfo(self.probs_dtype).tiny


    def _create_output(
            self,
            accepted: torch.Tensor,  # [batch_size, k]
            substitute_token_ids: torch.Tensor,  # [batch_size, k]
            draft_token_ids: torch.Tensor,  # [batch_size, k]
            bonus_token_ids: torch.Tensor = None,  # [batch_size] or [batch_size, 1] or None
            update_counter=False,
    ) -> torch.Tensor:
        """Format output. Returns a matrix of token ids. When
        a token is rejected via sampling, all subsequent token ids are 
        set to -1 for the sequence.

        输入形状:
            accepted: [batch_size, k] - bool tensor，表示是否接受每个草稿 token
            substitute_token_ids: [batch_size, k] - 拒绝时使用的替代 tokens
            draft_token_ids: [batch_size, k] - 草稿模型生成的 tokens
            bonus_token_ids: [batch_size] or [batch_size, 1] or None - bonus token（可选）

        输出形状:
            output: [batch_size, valid_len] - 接受的 tokens（-1 表示无效）
        """
        batch_size, k = substitute_token_ids.shape
        
        # Determine the index of the first False value for each row.
        # 如果 accepted 为空 (k=0)，直接返回空 output (不含 bonus token 也不可能生成，除非逻辑允许)
        if k == 0:
             return torch.empty((batch_size, 0), dtype=self.token_id_dtype, device=accepted.device)

        limits = (accepted == 0).max(1).indices
        limits[~(accepted == 0).any(1)] = k

        # Create masks using the indices.
        indices = torch.arange(k, device=accepted.device).unsqueeze(0)
        accepted_mask = indices < limits.unsqueeze(1)
        after_false_mask = indices == limits.unsqueeze(1)

        # Create an extended output tensor
        output_with_bonus_tokens = -torch.ones(
            (batch_size, k + self._num_bonus_tokens),
            dtype=self.token_id_dtype,
            device=accepted.device)
        output = output_with_bonus_tokens[:, :k]

        # Fill in the first k columns of the output tensor using masks and data
        # tensors.
        output[:, :k] = torch.where(accepted_mask, draft_token_ids,
                                    -torch.ones_like(draft_token_ids))

        # Fill the last column (bonus token).
        # We check output directly as accepted may have True values inconsistent
        # with causal acceptance.
        if bonus_token_ids is not None:
            bonus_token_ids = bonus_token_ids.squeeze(-1)  # 确保形状为 [batch_size]
            output_with_bonus_tokens[:, -1] = torch.where(output[:, -1] != -1,
                                                          bonus_token_ids, -1)
        else:
            # 没有 bonus token，最后一列保持为 -1
            output_with_bonus_tokens[:, -1] = -1

        # Fill the recovered token ids.
        output.mul_(~after_false_mask).add_(
            substitute_token_ids.mul(after_false_mask))

        
        if update_counter:
            self._ensure_counters(accepted.device)

            self.num_accepted_tokens.add_(accepted.sum())  # bool.sum -> long
            self.num_emitted_tokens.add_((output_with_bonus_tokens != -1).sum())
            self.num_draft_round.add_(batch_size)  # Python 标量可以直接加

        # bs = 1 时
        col_mask = (output_with_bonus_tokens[0] != -1)  # [k+1], bool
        output_with_bonus_tokens = output_with_bonus_tokens[:, col_mask]  # [1, valid_len]

        return output_with_bonus_tokens

    @torch.no_grad()
    def calculate_topk_entropy(self, logits, k):
        """
        高效地计算基于 Top-K 概率的近似熵。

        Args:
            logits (torch.Tensor): 模型输出的原始 logits，
                                形状为 [batch_size, seq_length, vocab_size]。
            k (int): 用于近似计算的 top k 值的数量。

        Returns:
            torch.Tensor: 每个 token 位置的近似熵，
                        形状为 [batch_size, seq_length]。
        """
        # 确保 k 值是有效的
        # vocab_size = logits.size(-1)
        # if k > vocab_size:
        #     raise ValueError(f"k ({k}) 不能大于词汇表大小 ({vocab_size})")

        # 1. 沿着最后一个维度（词汇表维度）计算概率分布
        #    使用 log_softmax 可以提高数值稳定性
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 2. 获取 Top-K 的 log 概率值
        #    我们只需要值，不需要它们的索引
        top_k_log_probs = torch.topk(log_probs, k, dim=-1).values

        # 3. 将 log 概率转换回普通概率
        top_k_probs = torch.exp(top_k_log_probs)

        # 4. 计算熵：H(p) = -Σ(p * log(p))
        #    由于我们已经有了 p 和 log(p)，可以直接计算
        #    p * log(p) 的部分，然后求和并取负
        entropy_terms = top_k_probs * top_k_log_probs
        top_k_entropy = -torch.sum(entropy_terms, dim=-1)

        return top_k_entropy

    def init_entropy_statistics(self):
        self.d_entropy_list = []
        self.t_entropy_list = []

    def entropy_statistics(self, accept_mask, draft_logits, target_logits, output=False):
        # accept_mask shape: [1, 25]
        # draft_logits/target_logits shape: [1, 25, vocab_size]

        # shape [1, 25]
        d_entropy = self.calculate_topk_entropy(draft_logits, 3)
        t_entropy = self.calculate_topk_entropy(target_logits, 3)

        # shape: [True_num]
        d_entropy_diff = d_entropy[~accept_mask]
        t_entropy_diff = t_entropy[~accept_mask]
        # self.cuslog.info(f"{t_entropy_diff=}")

        self.d_entropy_list.append(d_entropy_diff)
        self.t_entropy_list.append(t_entropy_diff)
        # self.cuslog.info(f"{self.t_entropy_list=}")

        if output:
            d_entropy_list = torch.cat(self.d_entropy_list, dim=0).view(-1)
            t_entropy_list = torch.cat(self.t_entropy_list, dim=0).view(-1)
            # self.cuslog.info(f"{t_entropy_list.shape=}")

            quantiles = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], device=accept_mask.device)
            d_values = torch.quantile(d_entropy_list.float(), quantiles)
            t_values = torch.quantile(t_entropy_list.float(), quantiles)

            self.cuslog.info(f"TotalNum:{d_entropy_list.shape[0]}, Draft entropy:{d_values}")
            self.cuslog.info(f"Target entropy: {t_values}")

    def rej_by_entropy(self, accept_mask, target_logits, thre):
        # shape [1, 25]
        t_entropy = self.calculate_topk_entropy(target_logits, 3)

        mask = (t_entropy < thre)

        rej_mask = ~(mask & (~accept_mask))

        # self.cuslog.info("="*50)
        # self.cuslog.info(f"{accept_mask=}")
        # self.cuslog.info(f"{rej_mask=}")
        # self.cuslog.info(f"{mask=}")
        # self.cuslog.info(f"{t_entropy=}")
        # self.cuslog.info("="*50)
        

        return rej_mask


    

    
            
    def ngram_find_candidate_pred_tokens(
        self,
        input_ids: torch.Tensor,
        max_ngram_size: int = 3,
        num_ngram_pred_tokens: int = 10,
    ):
        input_length = input_ids.size(1)

        for ngram_size in range(max_ngram_size, 0, -1):
            # 1. 构造所有窗口（去掉最后 n 个 token 以避免自匹配）
            if input_length <= ngram_size:        # 极短输入，无窗口
                continue
            windows = input_ids[:, :-ngram_size].unfold(1, ngram_size, 1)  # [B, W, n]
            # 2. 要匹配的 n-gram，扩成 [B, 1, n] 以便 broadcast
            ngram_tensor = input_ids[:, -ngram_size:].unsqueeze(1)         # [B, 1, n]
            # 3. 比对
            matches = (windows == ngram_tensor).all(-1)                    # [B, W]
            if not matches.any():
                continue

            # 4. 取第一处匹配（也可按其他策略挑选）
            window_idx = matches.nonzero(as_tuple=False)[0, 1]
            start_idx = window_idx + ngram_size
            end_idx = min(start_idx + num_ngram_pred_tokens, input_length)

            return input_ids[:, int(start_idx):int(end_idx)]

        return input_ids.new_empty((0,))
    
    def draft_with_ngram(self, prompt, past_kv_draft, temp):
        """
        使用 n-gram 匹配进行草稿生成
        
        Args:
            prompt: [1, seq_len] - 当前序列
            past_kv_draft: draft model 的 KV cache
            temp: 温度参数
        
        Returns:
            newly_gen_ids: [1, k] - 生成的 k 个 tokens
        """
        init_len = prompt.shape[1]
        # 初始化默认的 ngram draft（用于没有找到匹配时）
        ngram_draft_prev = torch.arange(101, self.num_ngram_pred_tokens + 101, dtype=prompt.dtype, device=prompt.device).unsqueeze(0)  # [1, num_ngram_pred_tokens]

        while prompt.shape[1] - init_len < self.k:

            kv_keep = prompt.shape[1] - self.bonus_tok_from_target
            past_kv_draft.crop(int(kv_keep))

            # 寻找 n-gram 匹配的候选 tokens
            ngram_draft = self.ngram_find_candidate_pred_tokens(prompt, self.max_ngram_size, self.num_ngram_pred_tokens)  # [1, num_ngram_pred_tokens]
            
            if ngram_draft.numel() > 0:
                ngram_draft_prev = ngram_draft
            else:
                ngram_draft = ngram_draft_prev

            num_ngram_pred_tokens = ngram_draft.shape[1]

            # 准备验证输入：最近的 bonus_tok_from_target 个 tokens + ngram draft
            input_verify = torch.cat([prompt[:,-self.bonus_tok_from_target:], ngram_draft], dim=1)  # [1, bonus_tok_from_target + num_ngram_pred_tokens]
            
            # Draft model forward
            out_d = self.draft_model(
                input_ids=input_verify,  # [1, bonus_tok_from_target + num_ngram_pred_tokens]
                use_cache=True,
                return_dict=True,
                past_key_values=past_kv_draft
            )

            # 获取 bonus token 的 logits
            bonus_tok_logits = out_d.logits[:,-1:,:]  # [1, 1, vocab_size]
            bonus_tok = sample_with_temperature(bonus_tok_logits, temp)  # [1, 1]

            # 验证 ngram draft tokens
            accepted, recovered_token_ids = self._batch_verification(
                    out_d.logits[:,-(num_ngram_pred_tokens+1):-1,:],  # [1, num_ngram_pred_tokens, vocab_size]
                    ngram_draft,  # [1, num_ngram_pred_tokens]
                    temp,
                )
            newly_ids = self._create_output(accepted, recovered_token_ids, ngram_draft, bonus_tok)  # [1, valid_len]
            
            # jz0805
            # newly_ids = out_d.logits[:,self.bonus_tok_from_target-1:self.bonus_tok_from_target,:].argmax(dim=-1)
            
            prompt = torch.cat([prompt, newly_ids], dim=1)


            # self.cuslog.info(f"jz0803 newly_ids in ngram1:{self.decode_ids(newly_ids)}")


            self.debug_ngram_accept_num.append(newly_ids.shape[1])

            self.bonus_tok_from_target = 1
        
        newly_gen_ids = prompt[:, init_len:]
        # prob = torch.cat(prob, dim=1)
        # self.cuslog.info(f"Before exit ngram, TotalLen:{prompt.shape[1]}, draft_kv_len:{past_kv_draft.get_seq_length()}")
        
        kv_keep = prompt.shape[1] - self.bonus_tok_from_target
        past_kv_draft.crop(int(kv_keep))

        return newly_gen_ids

    def expand_kv_cache(self, past_kv, batch_size):
        """
        将 batch_size=1 的 KV cache 扩展到 batch_size=N
        
        Args:
            past_kv: past_key_values，每层是 (key, value) tuple
                     key shape: [1, num_heads, seq_len, head_dim]
                     value shape: [1, num_heads, seq_len, head_dim]
            batch_size: 目标 batch size
        
        Returns:
            expanded_kv: batch_size=N 的 KV cache
                     key shape: [batch_size, num_heads, seq_len, head_dim]
                     value shape: [batch_size, num_heads, seq_len, head_dim]
        """
        if past_kv is None:
            return None
        
        expanded_kv = []
        for i, layer_kv in enumerate(past_kv):
            key, value = layer_kv  # key/value: [1, num_heads, seq_len, head_dim]
            # 在 batch 维度上复制，确保在相同设备上
            expanded_key = key.repeat(batch_size, 1, 1, 1)  # [batch_size, num_heads, seq_len, head_dim]
            expanded_value = value.repeat(batch_size, 1, 1, 1)  # [batch_size, num_heads, seq_len, head_dim]
            
            # 调试：检查设备
            if self.verbose and i == 0:
                self.cuslog.info(f"[expand_kv_cache] Layer 0: key device={key.device}, expanded_key device={expanded_key.device}")
            
            expanded_kv.append((expanded_key, expanded_value))
        
        # 返回与原始格式相同的结构
        return type(past_kv)(expanded_kv)
    
    def init_tree_mask(self, device):
        """初始化树形验证所需的掩码"""
        self.tree_mask_init = torch.eye(self.branch_n, device=device)[None, None]
        self.position_ids = torch.zeros(self.branch_n, device=device, dtype=torch.long)

    @torch.no_grad()
    def draft_with_tree(self, seed_token, past_kv_draft, temp=0, accepted_tokens=None):
        """
        树形草稿生成：逐层生成并剪枝的优化实现
        
        特点：
        1. 逐层生成，每层保留 max_nodes_per_level 个最高分节点
        2. 使用累积 log 概率作为 score
        3. 每个节点生成时输入完整路径（从 root 到当前节点）
        4. KV cache 管理：第一步时可以同时更新上一轮接受的 tokens
        
        Args:
            seed_token: [1, 1] or [1, n] 当前上下文的最后 token(s)
            past_kv_draft: draft model 的 past_key_values（原始上下文的 KV cache）
            temp: 温度参数（默认0，greedy）
            accepted_tokens: [1, m] 上一轮接受的 tokens（可选，用于更新 KV cache）
        
        Returns:
            tree_data: dict containing:
                - flat_tokens: torch.Tensor[T], 所有节点的 token_id
                - parent_indices: List[int], 每个节点的父节点索引
                - scores: torch.Tensor[T], 每个节点的累积得分
                - depths: List[int], 每个节点的深度
                - paths: List[List[int]], 从根到叶子的所有路径
                - updated_kv: 更新后的 KV cache（如果提供了 accepted_tokens）
        
        重要说明：
            - 如果提供了 accepted_tokens，会在第一步时一并输入以更新 KV cache
            - 这样可以避免在 tree verify 后再单独更新 KV cache
        """
        device = seed_token.device
        
        if self.verbose:
            self.cuslog.info(f"[draft_with_tree] Starting tree generation: k={self.k}, branch_n={self.branch_n}, max_nodes_per_level={self.max_nodes_per_level}")
        
        # 使用列表结构维护节点信息（未剪枝）
        score_list = []  # 所有节点的 score
        token_id_list = []  # 所有节点的 token_id
        parent_idx_list = []  # 所有节点的父节点索引（-1 表示根节点）
        
        # 剪枝后选择的节点索引
        selected_list = []  # 记录裁剪后选择的节点索引
        depth_list = []  # 记录 selected_list 中每个节点对应的深度（从1开始，与 depth 保持一致）
        
        # 第一步：生成第一层节点（depth=1）
        # 如果提供了 accepted_tokens，一并输入以更新 KV cache
        if accepted_tokens is not None and accepted_tokens.numel() > 0:
            # 输入 accepted_tokens + seed_token 以同时更新 KV cache
            first_input = torch.cat([accepted_tokens, seed_token], dim=1)  # [1, m+1]
        else:
            first_input = seed_token  # [1, 1]
        
        out = self.draft_model(
            input_ids=first_input,  # [1, 1] or [1, m+1]
            past_key_values=past_kv_draft,
            use_cache=True,
            return_dict=True,
        )
        past_kv_draft = out.past_key_values  # 更新 KV cache（现在包含了 accepted_tokens 和 seed_token）
        updated_kv = past_kv_draft  # 保存更新后的 KV cache
        
        # 保存当前 KV cache 的长度（包含 accepted_tokens 和 seed_token）
        # tree tokens 不会被加入 KV cache，每次 forward 后都会恢复到这个长度
        original_kv_len = past_kv_draft.get_seq_length() if past_kv_draft is not None else 0
        
        # 获取 logits 并计算 log probabilities
        logits = out.logits[:, -1, :]  # [1, vocab_size]
        if temp > 0:
            logits = logits / temp
        log_probs = F.log_softmax(logits, dim=-1)  # [1, vocab_size]
        
        # 获取 top-k 候选
        top_k_values, top_k_indices = torch.topk(log_probs[0], self.branch_n)
        
        # 创建第一层节点
        first_layer_indices = []
        for i in range(len(top_k_indices)):
            node_idx = len(score_list)  # 新节点的索引
            token_id_list.append(top_k_indices[i].item())
            score_list.append(top_k_values[i].item())  # 第一层的 score 就是 log_prob
            parent_idx_list.append(-1)  # 根节点没有父节点
            first_layer_indices.append(node_idx)
        
        # 剪枝第一层（如果需要）
        if len(first_layer_indices) > self.max_nodes_per_level:
            # 按 score 排序，保留前 max_nodes_per_level 个
            first_layer_indices.sort(key=lambda idx: score_list[idx], reverse=True)
            first_layer_indices = first_layer_indices[:self.max_nodes_per_level]
        
        # 将第一层选中的节点加入 selected_list，同时记录深度
        prev_selected_len = len(selected_list)  # extend 前的长度（应该是0）
        selected_list.extend(first_layer_indices)
        depth_list.extend([1] * len(first_layer_indices))  # 第一层的 depth=1
        
        if self.verbose:
            self.cuslog.info(f"[draft_with_tree] Layer 1: generated {len(first_layer_indices)} nodes")
        
        # 逐层扩展树（depth = 2 到 k）
        for depth in range(2, self.k + 1):
            # 获取当前层的节点索引：使用上一层的节点（通过 prev_selected_len 获取）
            current_layer_indices = selected_list[prev_selected_len:]
            
            # 将整个 selected_list 的所有 tokens 传入模型
            # 1. 收集 selected_list 中所有节点的 tokens
            all_tree_tokens = [token_id_list[idx] for idx in selected_list]
            
            # 2. 将 tokens 转换为 tensor
            num_tokens = len(selected_list)
            input_ids = torch.tensor(all_tree_tokens, device=device, dtype=torch.long).unsqueeze(0)  # [1, num_tokens]
            
            # 3. 获取 KV cache 长度
            kv_len = original_kv_len
            total_len = kv_len + num_tokens
            
            # 4. 构建 4D attention mask: [1, 1, num_tokens, total_len]
            # 初始化为 -inf（不能 attend）
            attention_mask = torch.full(
                (1, 1, num_tokens, total_len),
                float('-inf'),
                device=device,
                dtype=self.draft_model.dtype
            )
            
            # 4.1. Tree tokens 之间的 mask 构建
            # 为 selected_list 中的每个节点建立索引映射
            node_to_position = {node_idx: pos for pos, node_idx in enumerate(selected_list)}
            
            # 首先构建 tree tokens 之间的 mask [num_tokens, num_tokens]
            tree_mask = torch.zeros((num_tokens, num_tokens), device=device, dtype=torch.float32)
            
            # 从前往后迭代，利用父节点的 mask
            for i, node_idx in enumerate(selected_list):
                parent_node_idx = parent_idx_list[node_idx]
                
                if parent_node_idx == -1:
                    # 根节点：只能看到自己
                    tree_mask[i, i] = 1.0
                else:
                    # 子节点：继承父节点的 mask + 自己
                    parent_pos = node_to_position[parent_node_idx]
                    tree_mask[i, :] = tree_mask[parent_pos, :]
                    tree_mask[i, i] = 1.0
            
            # 4.2. 填充 attention_mask
            # 所有 Tree tokens 都可以看到完整的 KV cache
            attention_mask[:, :, :, :kv_len] = 0.0
            
            # Tree tokens 之间的可见性由 tree_mask 决定
            # tree_mask: [num_tokens, num_tokens] -> 1.0 表示可见
            # 我们需要将 1.0 映射为 0.0, 0.0 映射为 -inf
            mask_val = torch.where(tree_mask > 0, 0.0, float('-inf')).to(self.draft_model.dtype)
            attention_mask[:, :, :, kv_len:] = mask_val.unsqueeze(0).unsqueeze(0)
            
            # 5. 创建 position_ids
            position_ids = torch.arange(kv_len, kv_len + num_tokens, device=device, dtype=torch.long).unsqueeze(0)
            
            # 6. Forward 通过 draft model（直接传入 4D attention mask）
            out = self.draft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_kv_draft,
                use_cache=True,
                return_dict=True,
            )
            
            # 7. 提取 current_layer_indices 对应的 logits（它们在 selected_list 中的位置）
            all_logits = []
            for parent_idx in current_layer_indices:
                pos_in_selected = node_to_position[parent_idx]
                all_logits.append(out.logits[:, pos_in_selected:pos_in_selected+1, :])  # [1, 1, vocab_size]
            
            # 8. 恢复 KV cache 到原始长度（tree tokens 不应该被加入 KV cache）
            past_kv_draft = out.past_key_values
            past_kv_draft.crop(original_kv_len)
            
            # 批量处理所有节点的 logits
            all_logits = torch.cat(all_logits, dim=0)  # [num_nodes, 1, vocab_size]
            all_logits = all_logits.squeeze(1)         # [num_nodes, vocab_size]
            if temp > 0:
                all_logits = all_logits / temp
            log_probs = F.log_softmax(all_logits, dim=-1)  # [num_nodes, vocab_size]
            
            # 对每个父节点生成子节点
            next_layer_indices = []
            for i, parent_idx in enumerate(current_layer_indices):
                parent_score = score_list[parent_idx]
                
                # 获取该父节点的 top-k 候选
                node_log_probs = log_probs[i]  # [vocab_size]
                top_k_values, top_k_indices = torch.topk(node_log_probs, min(self.branch_n, node_log_probs.shape[-1]))
                
                # 创建子节点
                for j in range(len(top_k_indices)):
                    node_idx = len(score_list)  # 新节点的索引
                    token_id_list.append(top_k_indices[j].item())
                    score_list.append(parent_score + top_k_values[j].item())  # 累积 score
                    parent_idx_list.append(parent_idx)
                    next_layer_indices.append(node_idx)
            
            # 当前层剪枝：保留 score 最高的 max_nodes_per_level 个节点
            if len(next_layer_indices) > self.max_nodes_per_level:
                next_layer_indices.sort(key=lambda idx: score_list[idx], reverse=True)
                next_layer_indices = next_layer_indices[:self.max_nodes_per_level]
            
            # 将当前层选中的节点加入 selected_list，同时记录深度
            prev_selected_len = len(selected_list)  # 记录 extend 前的长度（即上一层的结束位置）
            selected_list.extend(next_layer_indices)
            depth_list.extend([depth] * len(next_layer_indices))  # 当前层的 depth（用于全局剪枝后）
            
            if self.verbose:
                self.cuslog.info(f"[draft_with_tree] Layer {depth}: generated {len(next_layer_indices)} nodes")
            
           
        
        # 恢复 past_kv_draft 到原始长度
        if past_kv_draft is not None:
            past_kv_draft.crop(original_kv_len)
        
        # 全局节点数限制（在 tree 构建完成后进行剪枝）
        if self.max_nodes_global and len(selected_list) > self.max_nodes_global:
            # 保留 score 最高的节点，同时保持 selected_list 和 depth_list 的对应关系
            node_scores = [(idx, score_list[idx], depth_list[i]) for i, idx in enumerate(selected_list)]
            node_scores.sort(key=lambda x: x[1], reverse=True)
            kept_nodes = node_scores[:self.max_nodes_global]
            selected_list = [x[0] for x in kept_nodes]
            depth_list = [x[2] for x in kept_nodes]  # 同时剪枝 depth_list
            
            if self.verbose:
                self.cuslog.info(f"[draft_with_tree] Global pruning: kept {len(selected_list)} nodes")
        
        if self.verbose:
            self.cuslog.info(f"[draft_with_tree] Tree generation completed: {len(selected_list)} selected nodes")
        
        # 1. 按照 depth_list 层数由浅到深排序 selected_list 和 depth_list
        # 创建 (depth, selected_idx, list_position) 的元组列表
        depth_sorted_data = [(depth_list[i], selected_list[i], i) for i in range(len(selected_list))]
        # 按 depth 排序（从小到大，即从浅到深）
        depth_sorted_data.sort(key=lambda x: x[0])
        
        # 提取排序后的 selected_list 和 depth_list
        sorted_selected_list = [x[1] for x in depth_sorted_data]
        sorted_depth_list = [x[0] for x in depth_sorted_data]
        
        
        # 2. 构建 attention_mask (tree_mask)
        # 为排序后的 selected_list 中的每个节点建立索引映射
        num_tokens = len(sorted_selected_list)
        node_to_position = {node_idx: pos for pos, node_idx in enumerate(sorted_selected_list)}
        
        # 构建 tree_mask：每个 token 可以看到它的所有 ancestors
        tree_mask = torch.zeros((num_tokens, num_tokens), device=device, dtype=torch.float32)
        
        # 从前往后迭代，利用父节点的 mask（父节点一定在子节点前面，因为按深度排序了）
        for i, node_idx in enumerate(sorted_selected_list):
            # 获取父节点
            parent_node_idx = parent_idx_list[node_idx]
            
            if parent_node_idx == -1:
                # 根节点：只能看到自己
                tree_mask[i, i] = 1.0
            else:
                # 子节点：继承父节点的 mask（父节点能看到的，子节点也能看到）
                if parent_node_idx in node_to_position:
                    parent_pos = node_to_position[parent_node_idx]
                    tree_mask[i, :] = tree_mask[parent_pos, :]
                # 再加上自己
                tree_mask[i, i] = 1.0
        
        # 添加 batch 和 head 维度，变成 [1, 1, num_tokens, num_tokens]
        attention_mask = tree_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, num_tokens, num_tokens]

        return {
            'score_list': score_list,
            'parent_idx_list': parent_idx_list,
            'token_id_list': token_id_list,
            'depth_list': sorted_depth_list,
            'selected_list': sorted_selected_list,
            'attention_mask': attention_mask,  # [1, 1, num_tokens, num_tokens]
            'updated_kv': updated_kv if accepted_tokens is not None else None  # 返回更新后的 KV cache
        }
    
    def build_tree_attention_mask(self, parent_indices, device):
        """
        构造 tree attention mask
        
        Args:
            parent_indices: List[int], 每个节点的父节点索引（-1 表示根节点）
            device: torch.device
        
        Returns:
            mask: [1, 1, T, T] float tensor，0.0 表示允许 attention，-inf 表示 mask
        """
        T = len(parent_indices)
        mask = torch.full((T, T), float('-inf'), device=device)
        
        for i in range(T):
            # 节点 i 可以看到自己及其所有祖先
            j = i
            while j != -1:
                mask[i, j] = 0.0
                if j < len(parent_indices):
                    j = parent_indices[j]
                else:
                    break
        
        # 添加 batch 和 head 维度
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        
        return mask
    
    def verify_tree_paths(
        self,
        tree_data: dict,
        target_logits: torch.Tensor,  # [1, T+1, vocab_size] 包含 seed logit
        temperature: float,
    ):
        """
        验证所有树路径，包含 FLy 算法逻辑
        
        Args:
            tree_data: draft_with_tree 的返回值
            target_logits: [1, T+1, vocab_size] target model 的完整 logits (包含 seed 的输出)
            temperature: 采样温度
        """
        # 移除 batch 维度，变为 [T+1, vocab_size]
        # 索引 0 对应 seed 的输出 (验证 Layer 1)
        # 索引 i+1 对应 tree_token[i] 的输出 (验证 children of tree_token[i])
        target_logits = target_logits.squeeze(0)
        device = target_logits.device
        
        selected_list = tree_data["selected_list"]
        parent_idx_list = tree_data["parent_idx_list"]
        token_id_list = tree_data["token_id_list"]
        
        T = len(selected_list)
        if T == 0:
            return None, None, None, None, -1
        
        # 1. 构建 draft_tokens [T]
        draft_tokens = torch.tensor([token_id_list[idx] for idx in selected_list], device=device)
        
        # 2. 确定每个 draft token 对应的 verification logit 索引
        # verification logit 是其父节点的输出
        node_to_position = {node_idx: pos for pos, node_idx in enumerate(selected_list)}
        logit_indices = []
        
        for node_idx in selected_list:
            parent_idx = parent_idx_list[node_idx]
            if parent_idx == -1:
                # 父节点是 Seed
                logit_indices.append(0)
            else:
                # 父节点是 tree token
                parent_pos = node_to_position[parent_idx]
                logit_indices.append(parent_pos + 1) # +1 因为 target_logits[0] 是 seed
                
        logit_indices = torch.tensor(logit_indices, device=device, dtype=torch.long)
        
        # 3. Gather 对应的 logits [T, vocab_size]
        verification_logits = target_logits[logit_indices]
        
        # 4. 采样 Target tokens
        if temperature == 0:
            target_tokens = verification_logits.argmax(dim=-1)
        else:
            probs = F.softmax(verification_logits / temperature, dim=-1)
            target_tokens = torch.multinomial(probs, 1).squeeze(-1)
            
        # 5. 比较得到基础 acceptance
        accepted_bool = (draft_tokens == target_tokens) # [T]
        
        # 6. 构建路径并应用 FLy
         # 找到所有叶子节点（没有子节点的节点）
        is_leaf = torch.ones(T, dtype=torch.bool, device=device)
        for i, node_idx in enumerate(selected_list):
            parent_idx = parent_idx_list[node_idx]
            if parent_idx != -1 and parent_idx in node_to_position:
                parent_pos = node_to_position[parent_idx]
                is_leaf[parent_pos] = False
        
        leaf_indices = torch.nonzero(is_leaf, as_tuple=True)[0].tolist()
        
        # 构建所有路径
        paths = []
        max_len = 0
        for leaf_pos in leaf_indices:
            path = []
            cur_pos = leaf_pos
            cur_node_idx = selected_list[cur_pos]
            while True:
                path.append(cur_pos)
                parent_idx = parent_idx_list[cur_node_idx]
                if parent_idx == -1:
                    break
                if parent_idx not in node_to_position:
                    break
                cur_pos = node_to_position[parent_idx]
                cur_node_idx = parent_idx
            path.reverse()
            paths.append(path)
            max_len = max(max_len, len(path))
            
        if len(paths) == 0:
            return None, None, None, None, -1

        # 构建 Path Tensors
        num_paths = len(paths)
        path_accepted = torch.zeros((num_paths, max_len), dtype=torch.bool, device=device)
        path_draft_tokens = torch.zeros((num_paths, max_len), dtype=torch.long, device=device)
        path_target_tokens = torch.zeros((num_paths, max_len), dtype=torch.long, device=device)
        path_valid_mask = torch.zeros((num_paths, max_len), dtype=torch.bool, device=device)
        
        for i, path in enumerate(paths):
            path_len = len(path)
            path_indices = torch.tensor(path, device=device, dtype=torch.long)
            path_accepted[i, :path_len] = accepted_bool[path_indices]
            path_draft_tokens[i, :path_len] = draft_tokens[path_indices]
            path_target_tokens[i, :path_len] = target_tokens[path_indices]
            path_valid_mask[i, :path_len] = True
            
        # 保存原始 accepted (用于统计)
        accepted_before_fly = path_accepted.clone() & path_valid_mask

        # 7. 应用 FLy 算法 (Pattern Matching)
        if self.enable_fly and max_len >= self.win_len:
            # self.pattern: [win_len] (False, True, True...)
            if not hasattr(self, 'pattern') or self.pattern.device != device:
                 self.pattern = torch.ones(self.win_len, dtype=torch.bool, device=device)
                 self.pattern[0] = False
            
            # Unfold path_accepted: [num_paths, num_windows, win_len]
            # 注意：unfold 会减少长度，只处理能完整覆盖窗口的部分
            if max_len >= self.win_len:
                # 为了处理变长路径，我们直接对整个 batch 操作，超出 valid_len 的部分是 False (0)，不会匹配全 1 (除非 pattern 全 0)
                unfold_accept = path_accepted.unfold(1, self.win_len, 1) 
                matched = torch.all(unfold_accept == self.pattern, dim=-1) # [num_paths, num_windows]
                
                updated_mask = torch.zeros_like(path_accepted, dtype=torch.bool, device=device)
                updated_mask[:, :matched.shape[1]] = matched
                
                # 更新 accepted: 将匹配位置 (False) 变为 True
                path_accepted = path_accepted | updated_mask
                
                # 保护逻辑: 最后 win_len 个位置不更新 (与原逻辑保持一致)
                # 注意：这里只保护 valid_len 的末尾，但 Tensor 是 padded 的。
                # 简化处理：只保护整个 tensor 的末尾（假设 max_len 足够长）
                # 更严谨的做法是按每个 path 的 length 保护，但为了效率，这里先使用全局保护
                # 或者我们只保护 matched 覆盖到的区域之后？
                # 按照 FLy 原始逻辑：updated_accept[:, -self.win_len:] = updated_accept[:, -self.win_len:] & accepted[:, -self.win_len:]
                path_accepted[:, -self.win_len:] = path_accepted[:, -self.win_len:] & accepted_before_fly[:, -self.win_len:]
                
        # 再次应用 valid mask 并计算 Causal Mask
        path_accepted = path_accepted & path_valid_mask
        cumulative_accepted = torch.cumprod(path_accepted.float(), dim=1).bool()
        final_accepted = cumulative_accepted
        
        # 8. 选择最佳路径
        accepted_lengths = final_accepted.sum(dim=1)
        best_path_id = accepted_lengths.argmax().item()
        best_len = accepted_lengths[best_path_id].item()
        
        # 提取最佳结果
        # 如果 best_len == 0，需要返回空 tensor 避免后续 shape 错误
        if best_len == 0:
             best_draft = torch.empty((1, 0), dtype=torch.long, device=device)
             best_target = torch.empty((1, 0), dtype=torch.long, device=device)
             best_accepted_mask = torch.empty((1, 0), dtype=torch.bool, device=device)
             best_accepted_before_fly_mask = torch.empty((1, 0), dtype=torch.bool, device=device)
        else:
            best_draft = path_draft_tokens[best_path_id, :best_len].unsqueeze(0)
            best_target = path_target_tokens[best_path_id, :best_len].unsqueeze(0)
            
            # 注意：这里返回的 accepted 应该是 final_accepted (processed by FLy and Causal)
            best_accepted_mask = final_accepted[best_path_id, :best_len].unsqueeze(0)
            best_accepted_before_fly_mask = accepted_before_fly[best_path_id, :best_len].unsqueeze(0)
        
        return (
            best_draft,
            best_accepted_mask,
            best_accepted_before_fly_mask,
            best_target,
            best_path_id
        )



    @torch.no_grad()
    def generate_chunks(self, input_ids, temperature):
        """
        主生成函数
        
        Args:
            input_ids: [1, seq_len] - 输入序列
            temperature: float - 采样温度
        
        Returns:
            outputs: [1, final_seq_len] - 生成的完整序列
        """
        init_len = input_ids.shape[1]
        self.reset_counter()

        input_ids = input_ids.to(self.draft_model.device)  # [1, seq_len]
        
        # Prefill 阶段
        out_d = self.draft_model(input_ids[:, :-1], use_cache=True, return_dict=True)
        past_kv_draft = out_d.past_key_values  # draft model 的 KV cache
        out_t = self.target_model(input_ids[:, :-1], use_cache=True, return_dict=True)
        past_kv_target = out_t.past_key_values  # target model 的 KV cache

        seed_for_next_d = input_ids[:, -1:]  # [1, 1] - draft 的下一个输入
        seed_for_next_t = input_ids[:, -1:]  # [1, 1] - target 的下一个输入
        
        self.start_time = time.time()
        if self.k == 0:
            outputs = self.target_model.generate(
                input_ids=input_ids,
                max_new_tokens=self.total_gen_tok,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                past_key_values=past_kv_target,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                )
            
            if self.verbose:
                self.cuslog.info(f">>>new output:{self.tokenizer.decode(outputs[0])}")
            # self.cuslog.info(f">>>generation_config:{self.target_model.generation_config.to_dict()}") 

            self.end_time = time.time()
            self.num_emitted_tokens = torch.tensor(outputs.shape[1] - init_len)
            self.num_draft_round = torch.tensor(outputs.shape[1] - init_len)
            self.num_accepted_tokens = 0

            return outputs


        draft_round = 0
        while (input_ids.shape[1] - init_len) < self.total_gen_tok:
            draft_round += 1

            if self.tree_verify:
                # Tree verify 模式：树形草稿生成 + 多路径验证
                if self.verbose:
                    self.cuslog.info(f"[Tree Verify] Round {draft_round}: Starting tree draft generation")
                
                # 准备树生成的输入
                # seed_for_next_d 可能包含 1 或 2 个 tokens（取决于 bonus_tok_from_target）
                # _last_accepted_tokens 是上一轮接受的 draft tokens（不含 bonus token）
                accepted_tokens_for_tree = None
                tree_seed = seed_for_next_d  # 默认使用 seed_for_next_d
                
                if hasattr(self, '_last_accepted_tokens') and self._last_accepted_tokens is not None:
                    # 有上一轮接受的 tokens，需要合并
                    if self.bonus_tok_from_target == 2:
                        # seed_for_next_d 包含了两个 tokens（最后一个接受的 + bonus）
                        # 我们需要：accepted_tokens + 这两个 tokens
                        tree_seed = seed_for_next_d[:, -1:]  # 只取最后一个作为 seed
                        accepted_tokens_for_tree = torch.cat([self._last_accepted_tokens, seed_for_next_d[:, :-1]], dim=1)
                    else:
                        # seed_for_next_d 只有一个 token（bonus）
                        # 上一轮的 accepted tokens 已经在 KV cache 中了
                        tree_seed = seed_for_next_d
                        accepted_tokens_for_tree = self._last_accepted_tokens
                
                # 生成树，同时更新 KV cache
                tree_data = self.draft_with_tree(
                    tree_seed, 
                    past_kv_draft, 
                    temp=temperature,
                    accepted_tokens=accepted_tokens_for_tree
                )
                
                past_kv_draft = tree_data['updated_kv']
                
                if self.verbose:
                    num_paths = len(tree_data.get('paths', []))
                    self.cuslog.info(f"[Tree Verify] Generated tree with {tree_data['flat_tokens'].numel()} nodes, {num_paths} paths")
                
                # 使用 tree attention 进行 target model forward
                # 从 tree_data 中提取所有 selected_list 的 tokens
                selected_list = tree_data['selected_list']
                token_id_list = tree_data['token_id_list']
                all_tree_tokens = [token_id_list[idx] for idx in selected_list]
                
                device = seed_for_next_t.device
                num_tokens = len(all_tree_tokens)
                
                # 将 tokens 转换为 tensor
                flat_tokens_tensor = torch.tensor(all_tree_tokens, device=device, dtype=torch.long).unsqueeze(0)  # [1, num_tokens]
                
                # 将 seed_for_next_t 的最后一个 token 和 flat_tokens 拼接
                # 只取最后一个 token，因为之前的 tokens 已经在 KV cache 中
                seed_token = seed_for_next_t[:, -1:]  # [1, 1]
                target_in = torch.cat([seed_token, flat_tokens_tensor], dim=1)  # [1, 1+num_tokens]
                
                # 获取 past_kv 的长度
                past_kv_len = past_kv_target.get_seq_length() if past_kv_target is not None else 0
                
                # 构建完整的 4D attention mask
                # tree_data['attention_mask'] 的形状是 [1, 1, num_tokens, num_tokens]
                tree_mask = tree_data['attention_mask']  # [1, 1, num_tokens, num_tokens]
                
                # 当前输入的长度 = 1 (seed) + num_tokens
                current_len = 1 + num_tokens
                total_len = past_kv_len + current_len
                
                # 构建完整的 4D attention mask: [1, 1, current_len, total_len]
                # current_len = 1 + num_tokens (seed + tree tokens)
                # total_len = past_kv_len + current_len
                full_attention_mask = torch.full(
                    (1, 1, current_len, total_len),
                    float('-inf'),
                    device=device,
                    dtype=self.target_model.dtype
                )
                
                # 1. Seed token (index 0 in current_len, index past_kv_len in total_len)
                # 可以看到所有 KV cache
                full_attention_mask[:, :, 0, :past_kv_len] = 0.0
                # 可以看到自己
                full_attention_mask[:, :, 0, past_kv_len] = 0.0
                
                # 2. Tree tokens (indices 1..num_tokens in current_len)
                # 可以看到所有 KV cache
                full_attention_mask[:, :, 1:, :past_kv_len] = 0.0
                # 可以看到 Seed token
                full_attention_mask[:, :, 1:, past_kv_len] = 0.0
                
                # Tree tokens 之间的可见性
                # tree_mask: [1, 1, num_tokens, num_tokens] -> 1.0 表示可见
                # 需要将 1.0 映射为 0.0, 0.0 映射为 -inf
                tree_mask_val = torch.where(tree_mask > 0, 0.0, float('-inf')).to(self.target_model.dtype)
                full_attention_mask[:, :, 1:, past_kv_len + 1:] = tree_mask_val
                
                # 创建 position_ids
                position_ids = torch.arange(past_kv_len, past_kv_len + current_len, device=device, dtype=torch.long).unsqueeze(0)
                
                # Target model forward（使用 tree attention mask）
                target_outputs = self.target_model(
                    input_ids=target_in,  # [1, 1+num_tokens]
                    attention_mask=full_attention_mask,  # [1, 1, current_len, total_len]
                    position_ids=position_ids,
                    past_key_values=past_kv_target,
                    use_cache=True,
                    return_dict=True,
                )
                
                # 提取 flat_tokens 对应的 logits（跳过 seed token）
                # target_logits_flat = target_outputs.logits[:, 1:, :]  # [1, T, vocab_size]
                # target_logits_flat = target_logits_flat.squeeze(0)  # [T, vocab_size]
                
                # Pass full logits (including seed output) to verify_tree_paths
                target_logits = target_outputs.logits  # [1, 1+T, vocab_size]
                
                # 在每条路径上执行 SPD+FLy，选择最佳路径
                best_draft_tokens, best_accepted, best_accepted_before_fly, best_recovered, best_path_id = self.verify_tree_paths(
                    tree_data, target_logits, temperature
                )
                
                
                # 调试：打印接受情况
                if self.verbose:
                    accepted_count = best_accepted.sum().item()
                    path_len = best_accepted.shape[1]
                    self.cuslog.info(f"[Tree Verify] Best path {best_path_id}: accepted {accepted_count}/{path_len} tokens")
                
                # 处理 bonus token
                # bonus token 只有在所有 draft tokens 都被接受时才存在
                # 此时 target_outputs.logits 的最后一个位置就是 bonus token 的 logits
                bonus_token_ids = None
                
                if best_accepted.all():
                    # 所有 draft tokens 都被接受，可以生成 bonus token
                    # target_outputs.logits 的形状是 [1, 1+T, vocab_size]
                    # 最后一个位置（索引 T）就是 bonus token 的 logits
                    bonus_logits = target_outputs.logits[:, -1:, :]  # [1, 1, vocab_size]
                    
                    # 采样 bonus token
                    if temperature == 0:
                        bonus_token_ids = bonus_logits.argmax(dim=-1)  # [1, 1]
                    else:
                        bonus_token_ids = sample_with_temperature(bonus_logits, temperature, 1)  # [1, 1]
                
                # 找到最后一个接受的位置（用于后续处理）
                
                else:
                    # 找到第一个 False 的位置
                    first_reject_idx = (best_accepted == 0).max(1).indices.item()
                
                # 记录统计信息
                if self.enable_statistics:
                    # 记录初始 mismatch 数量
                    initial_mismatch = (~best_accepted_before_fly).sum().item()
                    self.total_initial_mismatch += initial_mismatch
                    
                    # 记录 FLy 算法接受的数量
                    if self.enable_fly:
                        fly_accepted = (~best_accepted_before_fly) & best_accepted
                        self.total_fly_accepted += fly_accepted.sum().item()
                
                # 创建输出
                newly_input_ids = self._create_output(
                    best_accepted, best_recovered, best_draft_tokens, bonus_token_ids, update_counter=True
                )
                
                # === Tree Verify 全军覆没的补救措施 ===
                if newly_input_ids.shape[1] == 0:
                    # 使用 Target Model 对 Seed 的输出（真实 logits）生成一个 token
                    # target_logits[:, 0, :] 对应 Seed Token 的输出
                    seed_logits = target_logits[:, 0, :]
                    
                    if temperature == 0:
                        recovery_token = seed_logits.argmax(dim=-1, keepdim=True)
                    else:
                        recovery_token = sample_with_temperature(seed_logits.unsqueeze(1), temperature, 1)
                    
                    # 重新赋值 newly_input_ids，打破死循环
                    newly_input_ids = recovery_token
                    
                    # 既然全军覆没，重置 accepted 相关状态
                    draft_token_ids = recovery_token
                    # 这里的 accepted 全为 True (因为是 target model 自己生成的)
                    accepted = torch.ones_like(recovery_token, dtype=torch.bool)
                    
                    # 更新统计（如果需要的话，这里不算 mismatch，因为根本没 draft 成功）
                    if self.verbose:
                        self.cuslog.info("[Tree Verify] All paths rejected. Fallback to Target Model generation.")
                else:
                    # 正常更新 draft_token_ids 和 accepted 用于后续的 KV cache 管理
                    draft_token_ids = best_draft_tokens
                    accepted = best_accepted
                current_k = draft_token_ids.shape[1]
                
                if self.verbose:
                    self.cuslog.info(f"[Tree Verify] Draft tokens shape: {draft_token_ids.shape}, accepted shape: {accepted.shape}")
                
                # 验证完成后，更新 past_kv_draft
                # 注意：draft_with_tree 已经确保 past_kv_draft 回到了原始长度
                # 现在需要根据实际接受的情况更新 KV cache
                
                # _create_output 会处理接受的 tokens 并生成 newly_input_ids
                # 然后我们基于 newly_input_ids 来更新 KV cache
                # 暂时先不更新，等 newly_input_ids 生成后再更新
                
            elif self.use_ngram:
                draft_token_ids = self.draft_with_ngram(input_ids, past_kv_draft, 0)
                draft_token_ids = draft_token_ids[:,:self.k]

            else:
                draft_token_ids = []
                last = seed_for_next_d  # [1,1]
                for i in range(self.k):
                    
                    out_d = self.draft_model(input_ids=last, use_cache=True, return_dict=True, past_key_values=past_kv_draft)
                    next_token = sample_with_temperature(out_d.logits[:, -1:, :], 0)  # [1,1]
                    # next_token = out_d.logits[:, -1, :].argmax(dim=-1)
                    draft_token_ids.append(next_token)            # [1]
                    last = next_token

                    # self.cuslog.info(f"jz0803 without ngram output:{self.decode_ids(last)}")

                draft_token_ids = torch.cat(draft_token_ids, dim=1)  # [1, k]

            # 非 tree_verify 模式的原有逻辑
            if not self.tree_verify:
                # 将 seed token 和 draft tokens 拼接作为 target model 的输入
                # 只取 seed_for_next_t 的最后一个 token，因为之前的 tokens 已经在 KV cache 中
                seed_token = seed_for_next_t[:, -1:]  # [1, 1]
                target_in = torch.cat([seed_token, draft_token_ids], dim=1)  # [1, k+1]
                
                # Target model forward
                target_outputs = self.target_model(
                    input_ids=target_in,  # [1, k+1]
                    past_key_values=past_kv_target,
                    use_cache=True,
                    return_dict=True
                )
               
                current_k = draft_token_ids.shape[1]  # k 的实际值

                # 提取 target 在每个草稿位置的 logits（不含最后 bonus 位置）
                target_logits_k = target_outputs.logits[:, :-1, :][:, -current_k:, :]  # [1, k, vocab_size]

                # bonus token：最后一个位置的 logits
                bonus_logits = target_outputs.logits[:, -1:, :]  # [1, 1, vocab_size]
                # 采样 bonus token
                bonus_token_ids = sample_with_temperature(bonus_logits, temperature)[0]  # [1, 1]

                accepted, recovered_token_ids = self._batch_verification(
                    target_logits_k,        # [B, k, V]
                    draft_token_ids,        # [B, k]
                    temperature,
                )

                # 记录应用 FLy 算法前的 accepted 状态（用于统计）
                if self.enable_statistics:
                    accepted_before_fly = accepted.clone()

                # self.entropy_statistics(accepted, draft_probs, target_logits_k, output=(not (draft_round%10)))
                # self.cuslog.info(f"[{draft_round}]recovered_token_ids shape >>> {recovered_token_ids.shape}")
                # self.cuslog.info(f"[{draft_round}]accepted before >>> {accepted}")
                if self.entropy_thre:
                    rej_mask = self.rej_by_entropy(accepted, target_logits_k, self.entropy_thre)

                if self.enable_fly:
                    # 只有当序列长度 >= win_len 时才应用 FLy
                    if accepted.shape[1] >= self.win_len:
                        # FLy 算法：寻找 [False, True, ..., True] 模式并将 False 翻转为 True
                        self.pattern = torch.ones(self.win_len, dtype=torch.bool, device=accepted.device)  # [win_len]
                        self.pattern[0] = False  # 第一个位置为 False，其余为 True

                        # 使用滑动窗口展开 accepted
                        unfold_accept = accepted.unfold(1, self.win_len, 1)  # [1, num_windows, win_len]
                        # 检查每个窗口是否匹配模式
                        matched = torch.all(unfold_accept==self.pattern, dim=-1)  # [1, num_windows]

                        # 创建更新 mask
                        updated_mask = torch.zeros_like(accepted, dtype=torch.bool, device=accepted.device)  # [1, k]
                        updated_mask[:, :matched.shape[1]] = matched  # 将匹配的位置设为 True
                        updated_accept = accepted | updated_mask  # [1, k] - 将匹配位置的 False 翻转为 True

                        # 保护最后 win_len 个位置不被错误更新
                        updated_accept[:, -self.win_len:] = updated_accept[:, -self.win_len:] & accepted[:, -self.win_len:]
                        
                        # 统计指标：记录 FLy 算法接受的数量（在更新 accepted 之前）
                        if self.enable_statistics:
                            # 计算从 False 变成 True 的数量
                            fly_accepted = (~accepted_before_fly) & updated_accept
                            self.total_fly_accepted += fly_accepted.sum().item()
                        
                        accepted = updated_accept
                    # self.cuslog.info(f"[{draft_round}]accepted after >>> {accepted}")
                
                # 统计指标：记录初始 mismatch 数量（在应用 FLy 之前）
                if self.enable_statistics:
                    initial_mismatch = (~accepted_before_fly).sum().item()
                    self.total_initial_mismatch += initial_mismatch

                if self.entropy_thre:
                    accepted = accepted & rej_mask

                if self.abla_no_window:
                    accepted = rej_mask
                
                
                # shape [1, valid_len]
                newly_input_ids = self._create_output(
                        accepted,
                        recovered_token_ids,
                        draft_token_ids,
                        bonus_token_ids,
                        update_counter=True,
                    )
            
            
            input_ids = torch.cat([input_ids, newly_input_ids], dim=-1)
            # kv_keep = input_ids.shape[1] - 1

            # len_before_crop = past_kv_target.get_seq_length()
            # self.cuslog.info(f"[{draft_round}]past_kv_target Length >>> {past_kv_target.get_seq_length()}")
            # past_kv_target.crop(int(kv_keep))
            # past_kv_draft.crop(int(kv_keep))

            # 更新 KV cache（accepted 在所有路径中都已定义）
            # 注意：tree_verify 模式下，past_kv_draft 已在验证后正确更新，不需要再 crop
            if not self.tree_verify:
                # 非 tree_verify 模式：按原逻辑 crop KV cache
                if accepted.all():
                    self.bonus_tok_from_target = 2
                    past_kv_draft.crop(int(input_ids.shape[1]-2))
                    seed_for_next_d = input_ids[:,-2:]
                else:
                    self.bonus_tok_from_target = 1
                    past_kv_draft.crop(int(input_ids.shape[1]-1))
                    seed_for_next_d = input_ids[:,-1:]
            else:
                # tree_verify 模式下的 KV cache 管理
                newly_len = newly_input_ids.shape[1]
                if self.verbose:
                    self.cuslog.info(f"[Tree Verify] Generated {newly_len} new tokens in this round")
                
                # 计算实际接受的 token 数
                actual_accepted = 0
                if accepted.all():
                    actual_accepted = accepted.shape[1]  # 全部接受
                else:
                    # 找到第一个 False 的位置
                    first_false_idx = (accepted == 0).max(1).indices.item()
                    actual_accepted = first_false_idx
                
                # 保存接受的 tokens，供下一轮使用
                if actual_accepted > 0 and best_draft_tokens is not None:
                    self._last_accepted_tokens = best_draft_tokens[:, :actual_accepted]
                else:
                    self._last_accepted_tokens = None
                
                # 根据是否全部接受来决定下一轮的输入
                # 注意：KV cache 更新已经在 draft_with_tree 的第一步完成了
                if accepted.all():
                    # 全部接受，下一轮从最后两个 token 开始（包括 bonus token）
                    self.bonus_tok_from_target = 2
                    seed_for_next_d = input_ids[:, -2:]
                else:
                    # 部分接受或没有接受，下一轮从最后一个 token 开始
                    self.bonus_tok_from_target = 1
                    seed_for_next_d = input_ids[:, -1:]

            past_kv_target.crop(int(input_ids.shape[1] - 1))
            seed_for_next_t = input_ids[:,-1:]

            # self.cuslog.info(f"[{draft_round}]current mat >>> {newly_input_ids.shape[1]}")
            # self.cuslog.info(f"[{draft_round}]jz0803current round generate >>> {self.decode_ids(newly_input_ids)}")

            eos_found_mask = (newly_input_ids == self.tokenizer.eos_token_id) 
            eos_found_in_chunk = torch.any(eos_found_mask)
            if eos_found_in_chunk:
                break
        
        self.bonus_tok_from_target = 1
        self.end_time = time.time()


        # self.cuslog.info(f"[{draft_round}]jz0803 >>> {self.decode_ids(input_ids)}")

        return input_ids
    
    
    def show_status(self):
        elapsed = self.end_time - self.start_time

        cur_speed = self.num_emitted_tokens.item() / elapsed
        self.speed_list.append(cur_speed)
        speed = sum(self.speed_list) / len(self.speed_list)

        cur_mat = self.num_emitted_tokens.item() / self.num_draft_round.item()
        self.mat_list.append(cur_mat)
        mat = sum(self.mat_list) / len(self.mat_list)

        if len(self.debug_ngram_accept_num) > 0:
            mean_ngram_accept = sum(self.debug_ngram_accept_num) / len(self.debug_ngram_accept_num)
        else:
            mean_ngram_accept = 0

        status_msg = f"Speed:{speed:.2f}|MAT:{mat:.2f}|ngramMAT:{mean_ngram_accept:.2f}|DraftRound:{self.num_draft_round.cpu().item()}|TotalTok:{self.num_emitted_tokens}|Elapsed:{elapsed:.2f}"
        
        # 如果启用了统计，添加统计信息并累加到全局统计
        if self.enable_statistics:
            fly_accept_rate = (self.total_fly_accepted / self.total_initial_mismatch * 100) if self.total_initial_mismatch > 0 else 0.0
            total_final_rejected = self.total_initial_mismatch - self.total_fly_accepted
            status_msg += f"|InitialMismatch:{self.total_initial_mismatch}|FLyAccepted:{self.total_fly_accepted}|FinalRejected:{total_final_rejected}|FLyAcceptRate:{fly_accept_rate:.2f}%"
            
            # 将当前样本的统计累加到全局统计
            self.global_total_initial_mismatch += self.total_initial_mismatch
            self.global_total_fly_accepted += self.total_fly_accepted
            self.global_sample_count += 1
        
        self.cuslog.info(status_msg)
        
        result = {
            "accepted": self.num_accepted_tokens,
            "emitted": self.num_emitted_tokens,
            "draft_round": self.num_draft_round,
            "elapsed": elapsed,
        }
        
        # 如果启用了统计，添加统计结果
        if self.enable_statistics:
            result["initial_mismatch"] = self.total_initial_mismatch
            result["fly_accepted"] = self.total_fly_accepted
            result["final_rejected"] = self.total_initial_mismatch - self.total_fly_accepted
            result["fly_accept_rate"] = (self.total_fly_accepted / self.total_initial_mismatch * 100) if self.total_initial_mismatch > 0 else 0.0
        
        return result
    
    def get_statistics(self):
        """获取当前样本的统计指标"""
        if not self.enable_statistics:
            return None
        
        fly_accept_rate = (self.total_fly_accepted / self.total_initial_mismatch * 100) if self.total_initial_mismatch > 0 else 0.0
        total_final_rejected = self.total_initial_mismatch - self.total_fly_accepted
        
        return {
            "total_initial_mismatch": self.total_initial_mismatch,
            "total_fly_accepted": self.total_fly_accepted,
            "total_final_rejected": total_final_rejected,
            "fly_accept_rate": fly_accept_rate,
        }
    
    def get_global_statistics(self):
        """获取整个数据集的全局统计指标"""
        if not self.enable_statistics:
            return None
        
        # 计算全局总计
        global_fly_accept_rate = (self.global_total_fly_accepted / self.global_total_initial_mismatch * 100) if self.global_total_initial_mismatch > 0 else 0.0
        global_total_final_rejected = self.global_total_initial_mismatch - self.global_total_fly_accepted
        
        # 计算平均值（每个样本的平均值）
        avg_initial_mismatch = self.global_total_initial_mismatch / self.global_sample_count if self.global_sample_count > 0 else 0.0
        avg_fly_accepted = self.global_total_fly_accepted / self.global_sample_count if self.global_sample_count > 0 else 0.0
        avg_final_rejected = global_total_final_rejected / self.global_sample_count if self.global_sample_count > 0 else 0.0
        
        return {
            # 总计
            "global_total_initial_mismatch": self.global_total_initial_mismatch,
            "global_total_fly_accepted": self.global_total_fly_accepted,
            "global_total_final_rejected": global_total_final_rejected,
            "global_fly_accept_rate": global_fly_accept_rate,
            # 平均值
            "avg_initial_mismatch": avg_initial_mismatch,
            "avg_fly_accepted": avg_fly_accepted,
            "avg_final_rejected": avg_final_rejected,
            "avg_fly_accept_rate": global_fly_accept_rate,  # 全局接受率就是平均接受率
            # 样本数
            "sample_count": self.global_sample_count,
        }
    
    def print_global_statistics(self):
        """打印整个数据集的全局统计结果"""
        if not self.enable_statistics:
            return
        
        stats = self.get_global_statistics()
        if stats is None:
            return
        
        self.cuslog.info("=" * 80)
        self.cuslog.info("数据集全局统计结果 (Global Statistics)")
        self.cuslog.info("=" * 80)
        self.cuslog.info(f"处理的样本总数 (Total Samples): {stats['sample_count']}")
        self.cuslog.info("")
        self.cuslog.info("总计 (Totals):")
        self.cuslog.info(f"  初始 Mismatch 总数: {stats['global_total_initial_mismatch']}")
        self.cuslog.info(f"  FLy 接受总数: {stats['global_total_fly_accepted']}")
        self.cuslog.info(f"  最终拒绝总数: {stats['global_total_final_rejected']}")
        self.cuslog.info(f"  FLy 接受率: {stats['global_fly_accept_rate']:.2f}%")
        self.cuslog.info("")
        self.cuslog.info("平均值 (Averages per Sample):")
        self.cuslog.info(f"  平均初始 Mismatch: {stats['avg_initial_mismatch']:.2f}")
        self.cuslog.info(f"  平均 FLy 接受: {stats['avg_fly_accepted']:.2f}")
        self.cuslog.info(f"  平均最终拒绝: {stats['avg_final_rejected']:.2f}")
        self.cuslog.info(f"  平均 FLy 接受率: {stats['avg_fly_accept_rate']:.2f}%")
        self.cuslog.info("=" * 80)
    
    def reset_counter(self):
        self._counter_inited = False
        # 重置当前样本的统计变量（全局统计在 show_status 中累加，这里只重置当前样本的）
        if self.enable_statistics:
            self.total_initial_mismatch = 0
            self.total_fly_accepted = 0
        # 重置 tree_verify 相关状态
        if self.tree_verify:
            self._last_accepted_tokens = None
    
    def decode_ids(self, ids):
        token = self.tokenizer.decode(ids[0], skip_special_tokens=False)
        return token
