import torch
import time
import torch.nn.functional as F
from typing import Optional
from functools import cached_property



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
        self.max_nodes_per_level = spd_args.get("max_nodes_per_level", 100)  # 每层最多保留的节点数
        self.max_nodes_global = spd_args.get("max_nodes_global", None)  # 全局最多保留的节点数（None 表示不限制）
        
        # 在 tree_verify 模式下，限制 k 的最大值避免树过大
        if self.tree_verify:
            max_k_for_tree = 5  # 限制树的最大深度为 5
            if self.k > max_k_for_tree:
                self.cuslog.warning(f"[Tree Verify] k={self.k} is too large for tree mode, limiting to {max_k_for_tree}")
                self.k = max_k_for_tree
            
            # 设置合理的默认值
            if self.max_nodes_per_level > 50:
                self.max_nodes_per_level = 50  # 限制每层最多 50 个节点
            if self.max_nodes_global is None:
                self.max_nodes_global = 200  # 限制全局最多 200 个节点
            
            # 在 tree_verify 模式下强制启用 verbose 以便调试
            self.verbose = False
            self.cuslog.info(f"[Tree Verify] Enabled with k={self.k}, branch_n={self.branch_n}, max_nodes_per_level={self.max_nodes_per_level}, max_nodes_global={self.max_nodes_global}")
        
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
            bonus_token_ids: torch.Tensor,  # [batch_size] or [batch_size, 1]
            update_counter=False,
    ) -> torch.Tensor:
        """Format output. Returns a matrix of token ids. When
        a token is rejected via sampling, all subsequent token ids are 
        set to -1 for the sequence.

        输入形状:
            accepted: [batch_size, k] - bool tensor，表示是否接受每个草稿 token
            substitute_token_ids: [batch_size, k] - 拒绝时使用的替代 tokens
            draft_token_ids: [batch_size, k] - 草稿模型生成的 tokens
            bonus_token_ids: [batch_size] or [batch_size, 1] - bonus token

        输出形状:
            output: [batch_size, valid_len] - 接受的 tokens（-1 表示无效）
        """
        batch_size, k = substitute_token_ids.shape
        bonus_token_ids = bonus_token_ids.squeeze(-1)  # 确保形状为 [batch_size]
        # Determine the index of the first False value for each row.
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

        # Fill the last column.
        # We check output directly as accepted may have True values inconsistent
        # with causal acceptance.
        output_with_bonus_tokens[:, -1] = torch.where(output[:, -1] != -1,
                                                      bonus_token_ids, -1)

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
    def draft_with_tree(self, seed_token, past_kv_draft, temp=0):
        """
        树形草稿生成：采用类似 cnets.py 的高效批量处理方式
        
        关键：复用原始上下文的 KV cache（past_kv_draft），批量处理节点，
        使用 tree_mask 机制管理注意力。
        
        Args:
            seed_token: [1, 1] 当前上下文的最后一个 token
            past_kv_draft: draft model 的 past_key_values（原始上下文的 KV cache）
            temp: 温度参数
        
        Returns:
            tree_data: dict containing:
                - flat_tokens: torch.Tensor[T], 所有节点的 token_id
                - parent_indices: torch.Tensor[T], 每个节点的父节点索引
                - tree_mask: torch.Tensor[T, T], 树形注意力掩码
                - retrieve_indices: torch.Tensor[num_paths, max_depth], 从根到叶子的路径
                - tree_position_ids: torch.Tensor[T], 每个节点的位置ID
        """
        device = seed_token.device
        
        if self.verbose:
            self.cuslog.info(f"[draft_with_tree] Starting with device={device}, k={self.k}, branch_n={self.branch_n}")
        
        # 初始化树形掩码
        self.init_tree_mask(device)
        
        # 保存原始 KV cache 的长度
        original_kv_len = past_kv_draft.get_seq_length() if past_kv_draft is not None else 0
        len_posi = original_kv_len
        
        # 限制总节点数
        total_nodes = min(self.k * self.branch_n, self.max_nodes_global or 200)
        depth = self.k
        top_k = self.branch_n
        
        scores_list = []
        parents_list = []
        ss_token = []
        
        # 第一层：从 seed_token 生成 top-k 个候选
        out_hidden = self.draft_model(
            input_ids=seed_token,  # [1, 1]
            past_key_values=past_kv_draft,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = out_hidden.past_key_values
        
        # 获取 logits 并计算 log probabilities
        last_hidden = out_hidden.logits[:, -1]  # [1, vocab_size]
        if temp > 0:
            last_hidden = last_hidden / temp
        last_p = F.log_softmax(last_hidden, dim=-1)  # [1, vocab_size]
        
        # 获取 top-k
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values  # [1, top_k]
        scores = topk_p[0]  # [top_k]
        
        scores_list.append(scores[None])  # [1, top_k]
        parents_list.append(torch.zeros(1, dtype=torch.long, device=device))
        ss_token.append(topk_index[0])  # [top_k]
        
        input_ids = topk_index  # [1, top_k]
        input_hidden = out_hidden.hidden_states[-1][:, -1:, :] if hasattr(out_hidden, 'hidden_states') and out_hidden.hidden_states is not None else None
        
        # 如果没有 hidden_states，使用 embedding
        if input_hidden is None:
            # 获取 input embedding
            if hasattr(self.draft_model.model, 'embed_tokens'):
                input_embeds = self.draft_model.model.embed_tokens(topk_index)  # [1, top_k, hidden_size]
            else:
                # 如果模型结构不同，尝试其他方式
                input_embeds = self.draft_model.get_input_embeddings()(topk_index)  # [1, top_k, hidden_size]
            input_hidden = input_embeds.transpose(0, 1)  # [top_k, 1, hidden_size]
        else:
            input_hidden = input_hidden.repeat(top_k, 1, 1)  # [top_k, 1, hidden_size]
        
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=device)
        
        # 逐层扩展树（类似 cnets.py 的方式）
        for i in range(depth - 1):
            # 设置 tree_mask 用于注意力
            self.tree_mask = tree_mask
            position_ids = (len_posi + self.position_ids).unsqueeze(0)  # [1, top_k]
            
            # 批量处理当前层的所有节点
            # 注意：这里使用批量输入，而不是逐个处理
            out_hidden = self.draft_model(
                input_ids=input_ids,  # [1, top_k]
                past_key_values=past_key_values,
                position_ids=position_ids,  # [1, top_k]
                use_cache=True,
                return_dict=True,
                # 如果模型支持 attention_mask，可以传入
            )
            past_key_values = out_hidden.past_key_values
            len_posi += 1
            
            # 计算偏移量（用于 parent 索引）
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)
            
            # 获取 logits 并计算概率
            last_hidden = out_hidden.logits[:, -top_k:]  # [1, top_k, vocab_size]
            if temp > 0:
                last_hidden = last_hidden / temp
            last_p = F.log_softmax(last_hidden, dim=-1)  # [1, top_k, vocab_size]
            
            # 对每个节点获取 top-k 候选
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values  # [1, top_k, top_k]
            
            # 计算累积得分
            cu_scores = topk_p[0] + scores[:, None]  # [top_k, top_k]
            
            # 选择得分最高的 top_k 个节点
            topk_cs = torch.topk(cu_scores.view(-1), min(top_k, cu_scores.numel()), dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p  # 更新得分
            
            # 确定选中节点的父节点
            out_ids = topk_cs_index // top_k
            
            # 获取选中的 token ids
            input_ids = topk_index.view(-1)[topk_cs_index][None]  # [1, top_k]
            
            ss_token.append(input_ids[0])
            scores_list.append(cu_scores)
            
            # 更新 tree_mask
            if hasattr(out_hidden, 'hidden_states') and out_hidden.hidden_states is not None:
                input_hidden = out_hidden.hidden_states[-1][:, out_ids]
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
        
        # 整理最终的树结构（类似 cnets.py）
        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        
        # 选择得分最高的 total_nodes 个节点
        top_scores = torch.topk(scores_list, min(total_nodes, scores_list.numel()), dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        
        # 获取最终的 draft tokens
        draft_tokens = ss_token_list[top_scores_index]
        
        # 构建 parent indices
        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        
        # 构建 tree mask
        tree_mask = torch.eye(total_nodes, device=device).bool()
        tree_mask[:, 0] = True
        for i in range(total_nodes - 1):
            if i < len(mask_index_list):
                tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])
        
        # 计算 tree position ids
        tree_position_ids = torch.sum(tree_mask, dim=1) - 1
        
        # 转换为 float 并添加维度
        tree_mask = tree_mask.float()[None, None]
        
        # 构建 retrieve indices（从根到叶子的路径）
        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1 if noleaf_index else 0
        leaf_num = min(total_nodes - noleaf_num, total_nodes)
        
        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long, device=device) - 1
        retrieve_indices_list = retrieve_indices.tolist()
        
        rid = 0
        position_ids_list = tree_position_ids.tolist()
        
        for i in range(min(total_nodes, len(position_ids_list))):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    if rid < leaf_num and j < max_depth.item():
                        retrieve_indices_list[rid][j] = cid
                    if cid > 0 and cid - 1 < len(mask_index_list):
                        cid = mask_index_list[cid - 1]
                    else:
                        break
                rid += 1
                if rid >= leaf_num:
                    break
        
        retrieve_indices = torch.tensor(retrieve_indices_list, dtype=torch.long, device=device)
        
        # 裁剪 KV cache 回到原始长度
        if past_key_values is not None:
            past_key_values.crop(original_kv_len)
        
        return {
            'flat_tokens': draft_tokens,
            'parent_indices': mask_index,
            'tree_mask': tree_mask,
            'retrieve_indices': retrieve_indices,
            'tree_position_ids': tree_position_ids,
        }
    
    def build_tree_attention_mask(self, parent_index, device):
        """
        构造 tree attention mask
        
        Args:
            parent_index: List[int], 每个节点的父节点索引
            device: torch.device
        
        Returns:
            mask: [T, T] bool tensor, True 表示允许 attention，False 表示 mask
        """
        T = len(parent_index)
        mask = torch.zeros((T, T), dtype=torch.bool, device=device)
        
        for i in range(T):
            j = i
            while j != -1:
                mask[i, j] = True
                j = parent_index[j]
        
        return mask
    
    def verify_tree_paths(
        self,
        tree_data: dict,
        target_logits_flat: torch.Tensor,  # [T, vocab_size]
        temperature: float,
    ):
        """
        在每条路径上执行 SPD+FLy 验证，选择最佳路径
        
        Args:
            tree_data: draft_with_tree 返回的树数据
                - flat_tokens: torch.Tensor[T], 所有节点的 tokens
                - retrieve_indices: torch.Tensor[num_paths, max_depth], 路径索引
            target_logits_flat: [T, vocab_size] target model 对所有节点的 logits
            temperature: 温度参数
        
        Returns:
            best_path_tokens: [1, k_path] 最佳路径的 draft tokens
            best_path_accepted: [1, k_path] 最佳路径的 accepted mask（应用 FLy 后）
            best_path_accepted_before_fly: [1, k_path] 最佳路径的 accepted mask（应用 FLy 前）
            best_path_recovered: [1, k_path] 最佳路径的 recovered tokens  
            bonus_logits: [1, 1, vocab_size] 最后一个位置的 logits（用于 bonus token）
        """
        flat_tokens = tree_data['flat_tokens']
        retrieve_indices = tree_data['retrieve_indices']
        
        best_path_id = -1
        best_s_fly = -1
        best_path_data = None
        device = target_logits_flat.device
        
        # 对每条路径执行 SPD+FLy
        num_paths = retrieve_indices.shape[0]
        for path_id in range(num_paths):
            # 获取当前路径的索引（过滤掉 -1）
            path_indices = retrieve_indices[path_id]
            valid_mask = path_indices >= 0
            path_indices = path_indices[valid_mask]
            
            if len(path_indices) == 0:
                continue
            
            # 提取该路径的 draft tokens 和 target logits
            path_draft_tokens = flat_tokens[path_indices]
            path_target_logits = target_logits_flat[path_indices]
            
            # 确保是正确的形状
            if path_draft_tokens.dim() == 0:
                path_draft_tokens = path_draft_tokens.unsqueeze(0)
            draft_token_ids = path_draft_tokens.unsqueeze(0)  # [1, k_path]
            target_logits_k = path_target_logits.unsqueeze(0)  # [1, k_path, vocab_size]
            
            # SPD exact-match 验证
            accepted, recovered_token_ids = self._batch_verification(
                target_logits_k,
                draft_token_ids,
                temperature,
            )
            
            # 记录应用 FLy 前的状态（用于统计）
            accepted_before_fly = accepted.clone()
            
            # 应用熵门
            if self.entropy_thre:
                rej_mask = self.rej_by_entropy(accepted, target_logits_k, self.entropy_thre)
            
            # 应用 FLy 算法
            if self.enable_fly:
                # 只有当路径长度 >= win_len 时才应用 FLy
                if accepted.shape[1] >= self.win_len:
                    self.pattern = torch.ones(self.win_len, dtype=torch.bool, device=accepted.device)
                    self.pattern[0] = False
                    
                    unfold_accept = accepted.unfold(1, self.win_len, 1)
                    matched = torch.all(unfold_accept == self.pattern, dim=-1)
                    
                    updated_mask = torch.zeros_like(accepted, dtype=torch.bool, device=accepted.device)
                    updated_mask[:, :matched.shape[1]] = matched
                    updated_accept = accepted | updated_mask
                    
                    updated_accept[:, -self.win_len:] = updated_accept[:, -self.win_len:] & accepted[:, -self.win_len:]
                    
                    accepted = updated_accept
            
            if self.entropy_thre:
                accepted = accepted & rej_mask
            
            if self.abla_no_window:
                accepted = rej_mask
            
            # 计算该路径接受的 token 数
            s_fly = accepted.sum().item()
            
            # 选择 s_fly 最大的路径
            if s_fly > best_s_fly or (s_fly == best_s_fly and path_id == 0):
                best_s_fly = s_fly
                best_path_id = path_id
                best_path_data = {
                    'draft_token_ids': draft_token_ids,
                    'target_logits_k': target_logits_k,
                    'accepted': accepted,
                    'accepted_before_fly': accepted_before_fly,
                    'recovered_token_ids': recovered_token_ids,
                }
        
        if best_path_data is None:
            # 如果没有找到有效路径，返回空结果
            return None, None, None, None, None
        
        # 注意：tree verify 模式下，我们没有计算 bonus token 的 logits
        # 需要在主函数中单独计算 bonus logits
        return (
            best_path_data['draft_token_ids'],
            best_path_data['accepted'],
            best_path_data['accepted_before_fly'],
            best_path_data['recovered_token_ids'],
            None,  # 在 tree verify 模式下，bonus logits 需要单独计算
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

            # self.cuslog.info(f"[{draft_round}]jz0803ngram is {self.use_ngram} InputIds>>> {self.decode_ids(input_ids[:,-20:])}")

            if self.tree_verify:
                # Tree verify 模式：树形草稿生成 + 多路径验证
                if self.verbose:
                    self.cuslog.info(f"[Tree Verify] Starting tree draft generation with k={self.k}, branch_n={self.branch_n}")
                tree_data = self.draft_with_tree(seed_for_next_d, past_kv_draft, temp=0)
                
                # 检查树生成是否成功
                if tree_data['flat_tokens'].numel() == 0:
                    raise ValueError("Tree generation failed: no nodes generated. Check branch_n and model configuration.")
                
                if self.verbose:
                    num_paths = tree_data['retrieve_indices'].shape[0]
                    self.cuslog.info(f"[Tree Verify] Generated tree with {tree_data['flat_tokens'].numel()} nodes, {num_paths} paths")
                
                # 使用 tree attention 进行 target model forward
                flat_tokens = tree_data['flat_tokens']  # torch.Tensor[T]
                device = seed_for_next_t.device
                
                # 确保 flat_tokens 是正确的形状 [1, T]
                if flat_tokens.dim() == 1:
                    flat_tokens_tensor = flat_tokens.unsqueeze(0)  # [1, T]
                else:
                    flat_tokens_tensor = flat_tokens
                
                # 将 seed_for_next_t 和 flat_tokens 拼接
                target_in = torch.cat([seed_for_next_t, flat_tokens_tensor], dim=1)  # [1, 1+T]
                
                # 获取 past_kv 的长度
                past_kv_len = past_kv_target.get_seq_length() if past_kv_target is not None else 0
                
                # 当前输入的长度
                current_len = target_in.shape[1]  # 1 + T
                
                # 总序列长度 = past_kv_len + current_len
                total_seq_len = past_kv_len + current_len
                
                # 为了简化，我们不使用自定义的 attention_mask
                # 让模型使用默认的 causal mask，然后在验证阶段按路径处理
                # 这样可以避免形状不匹配的问题
                
                # Target model forward（不使用自定义 attention_mask）
                target_outputs = self.target_model(
                    input_ids=target_in,  # [1, 1+T]
                    past_key_values=past_kv_target,
                    use_cache=True,
                    return_dict=True,
                    # 不传入 attention_mask，使用默认的 causal mask
                )
                
                # 提取 flat_tokens 对应的 logits（跳过 seed token）
                target_logits_flat = target_outputs.logits[:, 1:, :]  # [1, T, vocab_size]
                target_logits_flat = target_logits_flat.squeeze(0)  # [T, vocab_size]
                
                # 在每条路径上执行 SPD+FLy，选择最佳路径
                best_draft_tokens, best_accepted, best_accepted_before_fly, best_recovered, _ = self.verify_tree_paths(
                    tree_data, target_logits_flat, temperature
                )
                
                # 检查是否找到有效路径
                if best_draft_tokens is None:
                    raise ValueError("Tree path verification failed: no valid path found. Check FLy and entropy threshold settings.")
                
                # 调试：打印接受情况
                if self.verbose:
                    accepted_count = best_accepted.sum().item()
                    self.cuslog.info(f"[Tree Verify] Path accepted {accepted_count}/{best_accepted.shape[1]} tokens, accepted mask: {best_accepted}")
                
                # 为 bonus token 计算 logits
                # 我们需要找到最后一个接受的位置，然后获取下一个位置的 logits
                # 注意：target_outputs.logits 的形状是 [1, 1+T, vocab_size]
                # 第0个位置对应seed token，第1到T个位置对应tree中的tokens
                
                # 找到最后一个接受的位置
                if best_accepted.all():
                    # 全部接受，bonus logits 在 target_outputs.logits 的最后一个位置后
                    # 但是我们没有计算那个位置！需要重新计算
                    last_accepted_idx = best_accepted.shape[1] - 1
                else:
                    # 找到第一个 False 的位置
                    first_reject_idx = (best_accepted == 0).max(1).indices.item()
                    last_accepted_idx = first_reject_idx - 1 if first_reject_idx > 0 else -1
                
                # 生成 bonus token
                # 为简化处理，我们直接使用 draft model 生成 bonus token
                # 这里我们用 seed_for_next_t 和接受的 tokens 来计算
                if last_accepted_idx >= 0:
                    # 有接受的 tokens
                    accepted_tokens = best_draft_tokens[:, :last_accepted_idx+1]  # [1, accepted_len]
                    bonus_input = torch.cat([seed_for_next_t, accepted_tokens], dim=1)  # [1, 1+accepted_len]
                else:
                    # 没有接受任何 token，只用 seed
                    bonus_input = seed_for_next_t  # [1, 1]
                
                # 使用 target model 计算 bonus logits
                bonus_out = self.target_model(
                    input_ids=bonus_input,
                    past_key_values=past_kv_target,
                    use_cache=False,  # 不更新 KV cache
                    return_dict=True
                )
                bonus_logits = bonus_out.logits[:, -1:, :]  # [1, 1, vocab_size]
                
                # 采样 bonus token
                if temperature == 0:
                    bonus_token_ids = bonus_logits.argmax(dim=-1)  # [1, 1]
                else:
                    bonus_token_ids = sample_with_temperature(bonus_logits, temperature, 1)  # [1, 1]
                
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
                
                # 更新 draft_token_ids 和 accepted 用于后续的 KV cache 管理
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
                target_in = torch.cat([seed_for_next_t, draft_token_ids], dim=1)  # [1, k+1]
                
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
                # tree_verify 模式：需要正确更新 KV cache
                # 根据 newly_input_ids 的长度来决定
                newly_len = newly_input_ids.shape[1]
                if self.verbose:
                    self.cuslog.info(f"[Tree Verify] Generated {newly_len} new tokens in this round")
                
                # tree_verify 模式下，需要更新 past_kv_draft
                # 计算实际接受的 draft token 数（不包括 bonus token）
                actual_draft_accepted = 0
                if accepted.all():
                    actual_draft_accepted = accepted.shape[1]  # 全部接受
                else:
                    # 找到第一个 False 的位置
                    first_false = (accepted == 0).max(1).indices.item()
                    actual_draft_accepted = first_false
                
                if self.verbose:
                    self.cuslog.info(f"[Tree Verify] Actual draft accepted: {actual_draft_accepted}, newly_len: {newly_len}")
                
                # 基于实际接受的 draft tokens 更新 KV cache
                if actual_draft_accepted > 0:
                    # 有接受的 draft tokens，更新 KV cache
                    # 使用接受的 draft tokens（不是 newly_input_ids，因为它可能包含 recovered token）
                    if best_draft_tokens is not None:
                        accepted_draft = best_draft_tokens[:, :actual_draft_accepted]
                        out_d = self.draft_model(
                            input_ids=accepted_draft,
                            use_cache=True,
                            return_dict=True,
                            past_key_values=past_kv_draft,
                        )
                        past_kv_draft = out_d.past_key_values
                    
                    # 如果接受了所有 draft tokens，bonus token 来自 target
                    if accepted.all():
                        self.bonus_tok_from_target = 2
                        seed_for_next_d = input_ids[:,-2:]
                    else:
                        self.bonus_tok_from_target = 1
                        seed_for_next_d = input_ids[:,-1:]
                else:
                    # 没有接受任何 draft token，past_kv_draft 保持不变
                    self.bonus_tok_from_target = 1
                    seed_for_next_d = input_ids[:,-1:]

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
    
    def decode_ids(self, ids):
        token = self.tokenizer.decode(ids[0], skip_special_tokens=False)
        return token
