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
    expanded_probs = probs.unsqueeze(1).expand(
        -1, num_samples, -1, -1
    )  # [batch_size, num_samples, k, vocab_size]
    
    q = torch.empty_like(expanded_probs)
    q.exponential_(1.0)
    
    samples = (expanded_probs / q).argmax(dim=-1)  # [batch_size, num_samples, k]
    
    return samples

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
    def __init__(self, draft_model, target_model, tokenizer, cuslog, spd_args):
        self.draft_model = draft_model.eval()
        self.target_model = target_model.eval()
        self.tokenizer = tokenizer
        self.cuslog = cuslog
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

        self.revise_decoding = spd_args.get("revise_decoding")
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

        # target 在每个草稿位置的 argmax
        # target_ids = target_logits.argmax(dim=-1)  # [B, k]
        if temp == 0:
            target_ids = sample_with_temperature(target_logits, temp) # [B, k]
        else:
            target_ids = sample_with_temperature(target_logits, temp, 2) # [B, k]

        # 接受：draft 等于 target 的 argmax
        accepted = (target_ids == draft_token_ids)  # [B, k], bool
        accepted = accepted.any(dim=0).unsqueeze(0)

        # 恢复 token：直接用 target 的 argmax（后续只在首次拒绝处写入）
        recovered_token_ids = target_ids[0:1,:]  # [B, k]

        
        

        return accepted, recovered_token_ids


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
        recovered_token_ids = _multinomial(
            recovered_probs,
            num_samples=1,
            k=k,
            seeded_seqs=seeded_seqs or {},
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
            bonus_token_ids: torch.Tensor,  # [batch_size]
            update_counter=False,
    ) -> torch.Tensor:
        """Format output. Returns a matrix of token ids. When
        a token is rejected via sampling, all subsequent token ids are 
        set to -1 for the sequence.

        Args:
            accepted: A boolean tensor indicating if the corresponding
            draft token in draft_token_ids should be accepted or not.
            substitute_token_ids: A tensor of token_ids that can be used
            as substitutes for the draft token ids if the proposed token
            is rejected.
            draft_token_ids: A tensor of token ids speculated by the 
            draft model.
            bonus_token_ids: Token ids to use as the bonus token if
            all the draft tokens are accepted.
        Returns:
            A tensor containing the accepted token ids. The shape of the 
            tensor is [batch_size, k + num_bonus_tokens]
        """
        batch_size, k = substitute_token_ids.shape
        bonus_token_ids = bonus_token_ids.squeeze(-1)
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
        init_len = prompt.shape[1]
        # prob = []
        ngram_draft_prev = torch.arange(101, self.num_ngram_pred_tokens + 101, dtype=prompt.dtype, device=prompt.device).unsqueeze(0) # jz0805


        while prompt.shape[1] - init_len < self.k:

            kv_keep = prompt.shape[1] - self.bonus_tok_from_target
            past_kv_draft.crop(int(kv_keep))

            # shape [1, num_ngram_pred_tokens]
            ngram_draft = self.ngram_find_candidate_pred_tokens(prompt, self.max_ngram_size, self.num_ngram_pred_tokens)
            
            if ngram_draft.numel() > 0:
                ngram_draft_prev = ngram_draft
            else:
                ngram_draft = ngram_draft_prev


            num_ngram_pred_tokens = ngram_draft.shape[1]

            input_verify = torch.cat([prompt[:,-self.bonus_tok_from_target:], ngram_draft], dim=1)
            
            out_d = self.draft_model(input_ids=input_verify, use_cache=True, return_dict=True, past_key_values=past_kv_draft)


            bonus_tok_logits = out_d.logits[:,-1:,:] #[bs,1,vocab_size]
            # bonus_tok = bonus_tok_logits.argmax(dim=-1) # [1,1]
            bonus_tok = sample_with_temperature(bonus_tok_logits, temp)

            accepted, recovered_token_ids = self._batch_verification(
                    out_d.logits[:,-(num_ngram_pred_tokens+1):-1,:], # [B, k, V]
                    ngram_draft,           # [B, k]
                    temp,
                )
            newly_ids = self._create_output(accepted, recovered_token_ids, ngram_draft, bonus_tok)
            
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



    @torch.no_grad()
    def generate_chunks(self, input_ids, temperature):
        init_len = input_ids.shape[1]
        self.reset_counter()
        # self.start_time = time.time()

        input_ids = input_ids.to(self.draft_model.device)  # jz0819
        # Prefill
        out_d = self.draft_model(input_ids[:, :-1], use_cache=True, return_dict=True)
        past_kv_draft = out_d.past_key_values
        out_t = self.target_model(input_ids[:, :-1], use_cache=True, return_dict=True)
        past_kv_target = out_t.past_key_values

        seed_for_next_d = input_ids[:, -1:]  # shape [1,1]
        seed_for_next_t = input_ids[:, -1:]
        
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
        self.cuslog.info("="*50)
        self.cuslog.info(f"InputIds>>> {self.decode_ids(input_ids)}")
        total_FLy_change = 0
        while (input_ids.shape[1] - init_len) < self.total_gen_tok:
            draft_round += 1

            
            if self.use_ngram:
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

            self.cuslog.info(f"[{draft_round}]DraftIds>>> {self.decode_ids(draft_token_ids)}")

            # self.cuslog.info(f"[{draft_round}] ngram is {self.use_ngram} Prompt>>> {self.decode_ids(input_ids)}")
            # self.cuslog.info(f"[{draft_round}]jz0803ngram is {self.use_ngram} NewlyGen>>> {self.decode_ids(draft_token_ids)}")

            target_in = torch.cat([seed_for_next_t, draft_token_ids], dim=1)  # [1, k+1]
            target_outputs = self.target_model(input_ids=target_in, past_key_values=past_kv_target, use_cache=True, return_dict=True)
           
            current_k = draft_token_ids.shape[1]

            # self.cuslog.info(f"[{draft_round}] current_k is {current_k}")


            # target 每个草稿位置的 logits（不含最后 bonus 位置）
            # target_logits_k = target_outputs.logits[:, -(current_k+1):-1, :]  # [B, k, V]
            target_logits_k = target_outputs.logits[:, :-1, :][:, -current_k:, :]

            # bonus token：最后一步的 argmax
            bonus_logits = target_outputs.logits[:, -1:, :]       # [B, V]
            # bonus_token_ids = bonus_logits.argmax(dim=-1)        # [B]
            bonus_token_ids = sample_with_temperature(bonus_logits, temperature)[0]

            accepted, recovered_token_ids = self._batch_verification(
                target_logits_k,        # [B, k, V]
                draft_token_ids,        # [B, k]
                temperature,
            )

            # self.entropy_statistics(accepted, draft_probs, target_logits_k, output=(not (draft_round%10)))
            # self.cuslog.info(f"[{draft_round}]recovered_token_ids shape >>> {recovered_token_ids.shape}")
            
            self.cuslog.info(f"[{draft_round}]TargetIds>>> {self.decode_ids(recovered_token_ids)}")
            self.cuslog.info(f"[{draft_round}]accepted without FLy >>> {accepted}")
            previous_accept = accepted.clone()

            if self.entropy_thre:
                rej_mask = self.rej_by_entropy(accepted, target_logits_k, self.entropy_thre)

            if self.revise_decoding:
                self.pattern = torch.ones(self.win_len, dtype=torch.bool, device=accepted.device)
                self.pattern[0] = False

                unfold_accept = accepted.unfold(1, self.win_len, 1)
                matched = torch.all(unfold_accept==self.pattern, dim=-1)

                updated_mask = torch.zeros_like(accepted, dtype=torch.bool, device=accepted.device)
                updated_mask[:, :matched.shape[1]] = matched
                updated_accept = accepted | updated_mask

                updated_accept[:, -self.win_len:] = updated_accept[:, -self.win_len:] & accepted[:, -self.win_len:]
                
                accepted = updated_accept
                # self.cuslog.info(f"[{draft_round}]accepted after >>> {accepted}")

            if self.entropy_thre:
                accepted = accepted & rej_mask

            self.cuslog.info(f"[{draft_round}]accepted with FLy >>> {accepted}")
            
            diff_count = torch.ne(previous_accept, accepted).sum().item()
            total_FLy_change += diff_count
            
            # shape [1, valid_len]
            newly_input_ids = self._create_output(
                    accepted,
                    recovered_token_ids,
                    draft_token_ids,
                    bonus_token_ids,
                    update_counter=True,
                )
            
            self.cuslog.info(f"[{draft_round}]NewlyGeneratedIds>>> {self.decode_ids(newly_input_ids)}")
            
            input_ids = torch.cat([input_ids, newly_input_ids], dim=-1)
            # kv_keep = input_ids.shape[1] - 1

            # len_before_crop = past_kv_target.get_seq_length()
            # self.cuslog.info(f"[{draft_round}]past_kv_target Length >>> {past_kv_target.get_seq_length()}")
            # past_kv_target.crop(int(kv_keep))
            # past_kv_draft.crop(int(kv_keep))

            if accepted.all():
                self.bonus_tok_from_target = 2
                past_kv_draft.crop(int(input_ids.shape[1]-2))
                seed_for_next_d = input_ids[:,-2:]
            else:
                self.bonus_tok_from_target = 1
                past_kv_draft.crop(int(input_ids.shape[1]-1))
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

        if total_FLy_change > 10:
            self.cuslog.info(f"[{draft_round}]FLy change {total_FLy_change} tokens >>> {self.decode_ids(input_ids)}")
        else:
            self.cuslog.info(f"{total_FLy_change=}")

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

        self.cuslog.info(f"Speed:{speed:.2f}|MAT:{mat:.2f}|ngramMAT:{mean_ngram_accept:.2f}|DraftRound:{self.num_draft_round.cpu().item()}|TotalTok:{self.num_emitted_tokens}|Elapsed:{elapsed:.2f}")
        
        return {
            "accepted": self.num_accepted_tokens,
            "emitted": self.num_emitted_tokens,
            "draft_round": self.num_draft_round,
            "elapsed": elapsed,
        }
    
    def reset_counter(self):
        self._counter_inited = False
    
    def decode_ids(self, ids):
        token = self.tokenizer.decode(ids[0], skip_special_tokens=False)
        return token
