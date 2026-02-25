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
    expanded_probs = probs.unsqueeze(1).expand(
        -1, num_samples, -1, -1
    )  # [batch_size, num_samples, k, vocab_size]
    
    q = torch.empty_like(expanded_probs)  # [batch_size, num_samples, k, vocab_size]
    q.exponential_(1.0)
    
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

        self._counter_inited=False

        self.cuslog.info(f"{self.draft_model.device=}, {self.target_model.device=}, {self.target_model.dtype=}")
        self.speed_list, self.mat_list = [], []

        self.enable_fly = spd_args.get("enable_fly")
        self.win_len = spd_args.get("win_len")

        entropy_thre = float(spd_args.get("entropy_thre", 0))
        self.entropy_thre = entropy_thre if entropy_thre > 1e-2 else None

        self.use_ngram       = spd_args.get("use_ngram", False)
        self.max_ngram_size  = spd_args.get("max_ngram_size", 3)
        self.num_ngram_pred_tokens = spd_args.get("num_ngram_pred_tokens", 10)

        self.debug_ngram_accept_num = []
        self.bonus_tok_from_target = 1

        self.verbose = spd_args.get("verbose", False)

        self.abla_no_window = spd_args.get("abla_no_window", False)

        self.enable_statistics = spd_args.get("enable_statistics", False)
        
        self.tree_verify = spd_args.get("tree_verify", False)
        self.branch_n = spd_args.get("branch_n", 10)
        self.max_nodes_per_level = spd_args.get("max_nodes_per_level", 10)
        self.max_nodes_global = spd_args.get("max_nodes_global", 100)
        
        if self.tree_verify:
            self.use_ngram = False
        
        self.total_initial_mismatch = 0
        self.total_fly_accepted = 0
        
        self.global_total_initial_mismatch = 0
        self.global_total_fly_accepted = 0
        self.global_sample_count = 0



    def _ensure_counters(self, device: torch.device):
        if self._counter_inited:
            if self.num_accepted_tokens.device != device:
                self.num_accepted_tokens = self.num_accepted_tokens.to(device, non_blocking=True)
                self.num_emitted_tokens  = self.num_emitted_tokens.to(device, non_blocking=True)
                self.num_draft_round    = self.num_draft_round.to(device, non_blocking=True)
            return

        self.num_accepted_tokens = torch.zeros((), dtype=torch.long, device=device)
        self.num_emitted_tokens  = torch.zeros((), dtype=torch.long, device=device)
        self.num_draft_round    = torch.zeros((), dtype=torch.long, device=device)
        self._counter_inited = True


    def _batch_verification(
        self,
        target_logits: torch.Tensor,
        draft_token_ids: torch.Tensor, # [B, k]
        temp=0,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if temp == 0:
            target_ids = sample_with_temperature(target_logits, temp)  # [1, k] (sample_times=1)
        else:
            target_ids = sample_with_temperature(target_logits, temp, 2)  # [2, k] (sample_times=2)

        accepted = (target_ids == draft_token_ids)  # [sample_times, k], bool
        accepted = accepted.any(dim=0).unsqueeze(0)

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
        
        selected_draft_probs = torch.gather(draft_probs, dim=-1, index=draft_token_ids.unsqueeze(-1)).squeeze(-1)

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
        batch_size, k = substitute_token_ids.shape
        
        # Determine the index of the first False value for each row.
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
            bonus_token_ids = bonus_token_ids.squeeze(-1)
            output_with_bonus_tokens[:, -1] = torch.where(output[:, -1] != -1,
                                                          bonus_token_ids, -1)
        else:
            output_with_bonus_tokens[:, -1] = -1

        # Fill the recovered token ids.
        output.mul_(~after_false_mask).add_(
            substitute_token_ids.mul(after_false_mask))

        
        if update_counter:
            self._ensure_counters(accepted.device)

            self.num_accepted_tokens.add_(accepted.sum())  # bool.sum -> long
            self.num_emitted_tokens.add_((output_with_bonus_tokens != -1).sum())
            self.num_draft_round.add_(batch_size)

        col_mask = (output_with_bonus_tokens[0] != -1)  # [k+1], bool
        output_with_bonus_tokens = output_with_bonus_tokens[:, col_mask]  # [1, valid_len]

        return output_with_bonus_tokens

    @torch.no_grad()
    def calculate_topk_entropy(self, logits, k):
        # vocab_size = logits.size(-1)
        # if k > vocab_size:

        log_probs = F.log_softmax(logits, dim=-1)
        
        top_k_log_probs = torch.topk(log_probs, k, dim=-1).values

        top_k_probs = torch.exp(top_k_log_probs)

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

        self.d_entropy_list.append(d_entropy_diff)
        self.t_entropy_list.append(t_entropy_diff)

        if output:
            d_entropy_list = torch.cat(self.d_entropy_list, dim=0).view(-1)
            t_entropy_list = torch.cat(self.t_entropy_list, dim=0).view(-1)

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

        return rej_mask


    

    
            
    def ngram_find_candidate_pred_tokens(
        self,
        input_ids: torch.Tensor,
        max_ngram_size: int = 3,
        num_ngram_pred_tokens: int = 10,
    ):
        input_length = input_ids.size(1)

        for ngram_size in range(max_ngram_size, 0, -1):
            if input_length <= ngram_size:
                continue
            windows = input_ids[:, :-ngram_size].unfold(1, ngram_size, 1)  # [B, W, n]
            ngram_tensor = input_ids[:, -ngram_size:].unsqueeze(1)         # [B, 1, n]
            matches = (windows == ngram_tensor).all(-1)                    # [B, W]
            if not matches.any():
                continue

            window_idx = matches.nonzero(as_tuple=False)[0, 1]
            start_idx = window_idx + ngram_size
            end_idx = min(start_idx + num_ngram_pred_tokens, input_length)

            return input_ids[:, int(start_idx):int(end_idx)]

        return input_ids.new_empty((0,))
    
    def draft_with_ngram(self, prompt, past_kv_draft, temp):
        init_len = prompt.shape[1]
        ngram_draft_prev = torch.arange(101, self.num_ngram_pred_tokens + 101, dtype=prompt.dtype, device=prompt.device).unsqueeze(0)  # [1, num_ngram_pred_tokens]

        while prompt.shape[1] - init_len < self.k:

            kv_keep = prompt.shape[1] - self.bonus_tok_from_target
            past_kv_draft.crop(int(kv_keep))

            ngram_draft = self.ngram_find_candidate_pred_tokens(prompt, self.max_ngram_size, self.num_ngram_pred_tokens)  # [1, num_ngram_pred_tokens]
            
            if ngram_draft.numel() > 0:
                ngram_draft_prev = ngram_draft
            else:
                ngram_draft = ngram_draft_prev

            num_ngram_pred_tokens = ngram_draft.shape[1]

            input_verify = torch.cat([prompt[:,-self.bonus_tok_from_target:], ngram_draft], dim=1)  # [1, bonus_tok_from_target + num_ngram_pred_tokens]
            
            # Draft model forward
            out_d = self.draft_model(
                input_ids=input_verify,  # [1, bonus_tok_from_target + num_ngram_pred_tokens]
                use_cache=True,
                return_dict=True,
                past_key_values=past_kv_draft
            )

            bonus_tok_logits = out_d.logits[:,-1:,:]  # [1, 1, vocab_size]
            bonus_tok = sample_with_temperature(bonus_tok_logits, temp)  # [1, 1]

            accepted, recovered_token_ids = self._batch_verification(
                    out_d.logits[:,-(num_ngram_pred_tokens+1):-1,:],  # [1, num_ngram_pred_tokens, vocab_size]
                    ngram_draft,  # [1, num_ngram_pred_tokens]
                    temp,
                )
            newly_ids = self._create_output(accepted, recovered_token_ids, ngram_draft, bonus_tok)  # [1, valid_len]
            
            prompt = torch.cat([prompt, newly_ids], dim=1)


            self.debug_ngram_accept_num.append(newly_ids.shape[1])

            self.bonus_tok_from_target = 1
        
        newly_gen_ids = prompt[:, init_len:]
        
        kv_keep = prompt.shape[1] - self.bonus_tok_from_target
        past_kv_draft.crop(int(kv_keep))

        return newly_gen_ids

    def expand_kv_cache(self, past_kv, batch_size):
        if past_kv is None:
            return None
        
        expanded_kv = []
        for i, layer_kv in enumerate(past_kv):
            key, value = layer_kv  # key/value: [1, num_heads, seq_len, head_dim]
            expanded_key = key.repeat(batch_size, 1, 1, 1)  # [batch_size, num_heads, seq_len, head_dim]
            expanded_value = value.repeat(batch_size, 1, 1, 1)  # [batch_size, num_heads, seq_len, head_dim]
            
            if self.verbose and i == 0:
                self.cuslog.info(f"[expand_kv_cache] Layer 0: key device={key.device}, expanded_key device={expanded_key.device}")
            
            expanded_kv.append((expanded_key, expanded_value))
        
        return type(past_kv)(expanded_kv)
    
    def init_tree_mask(self, device):
        self.tree_mask_init = torch.eye(self.branch_n, device=device)[None, None]
        self.position_ids = torch.zeros(self.branch_n, device=device, dtype=torch.long)

    @torch.no_grad()
    def draft_with_tree(self, seed_token, past_kv_draft, temp=0, accepted_tokens=None):
        device = seed_token.device
        
        if self.verbose:
            self.cuslog.info(f"[draft_with_tree] Starting tree generation: k={self.k}, branch_n={self.branch_n}, max_nodes_per_level={self.max_nodes_per_level}")
        
        score_list = []
        token_id_list = []
        parent_idx_list = []
        
        selected_list = []
        depth_list = []
        
        if accepted_tokens is not None and accepted_tokens.numel() > 0:
            first_input = torch.cat([accepted_tokens, seed_token], dim=1)  # [1, m+1]
        else:
            first_input = seed_token  # [1, 1]
        
        out = self.draft_model(
            input_ids=first_input,  # [1, 1] or [1, m+1]
            past_key_values=past_kv_draft,
            use_cache=True,
            return_dict=True,
        )
        past_kv_draft = out.past_key_values
        updated_kv = past_kv_draft
        
        original_kv_len = past_kv_draft.get_seq_length() if past_kv_draft is not None else 0
        
        logits = out.logits[:, -1, :]  # [1, vocab_size]
        if temp > 0:
            logits = logits / temp
        log_probs = F.log_softmax(logits, dim=-1)  # [1, vocab_size]
        
        top_k_values, top_k_indices = torch.topk(log_probs[0], self.branch_n)
        
        first_layer_indices = []
        for i in range(len(top_k_indices)):
            node_idx = len(score_list)
            token_id_list.append(top_k_indices[i].item())
            score_list.append(top_k_values[i].item())
            parent_idx_list.append(-1)
            first_layer_indices.append(node_idx)
        
        if len(first_layer_indices) > self.max_nodes_per_level:
            first_layer_indices.sort(key=lambda idx: score_list[idx], reverse=True)
            first_layer_indices = first_layer_indices[:self.max_nodes_per_level]
        
        prev_selected_len = len(selected_list)
        selected_list.extend(first_layer_indices)
        depth_list.extend([1] * len(first_layer_indices))
        
        if self.verbose:
            self.cuslog.info(f"[draft_with_tree] Layer 1: generated {len(first_layer_indices)} nodes")
        
        for depth in range(2, self.k + 1):
            current_layer_indices = selected_list[prev_selected_len:]
            
            all_tree_tokens = [token_id_list[idx] for idx in selected_list]
            
            num_tokens = len(selected_list)
            input_ids = torch.tensor(all_tree_tokens, device=device, dtype=torch.long).unsqueeze(0)  # [1, num_tokens]
            
            kv_len = original_kv_len
            total_len = kv_len + num_tokens
            
            attention_mask = torch.full(
                (1, 1, num_tokens, total_len),
                float('-inf'),
                device=device,
                dtype=self.draft_model.dtype
            )
            
            node_to_position = {node_idx: pos for pos, node_idx in enumerate(selected_list)}
            
            tree_mask = torch.zeros((num_tokens, num_tokens), device=device, dtype=torch.float32)
            
            for i, node_idx in enumerate(selected_list):
                parent_node_idx = parent_idx_list[node_idx]
                
                if parent_node_idx == -1:
                    tree_mask[i, i] = 1.0
                else:
                    parent_pos = node_to_position[parent_node_idx]
                    tree_mask[i, :] = tree_mask[parent_pos, :]
                    tree_mask[i, i] = 1.0
            
            attention_mask[:, :, :, :kv_len] = 0.0
            
            mask_val = torch.where(tree_mask > 0, 0.0, float('-inf')).to(self.draft_model.dtype)
            attention_mask[:, :, :, kv_len:] = mask_val.unsqueeze(0).unsqueeze(0)
            
            position_ids = torch.arange(kv_len, kv_len + num_tokens, device=device, dtype=torch.long).unsqueeze(0)
            
            out = self.draft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_kv_draft,
                use_cache=True,
                return_dict=True,
            )
            
            all_logits = []
            for parent_idx in current_layer_indices:
                pos_in_selected = node_to_position[parent_idx]
                all_logits.append(out.logits[:, pos_in_selected:pos_in_selected+1, :])  # [1, 1, vocab_size]
            
            past_kv_draft = out.past_key_values
            past_kv_draft.crop(original_kv_len)
            
            all_logits = torch.cat(all_logits, dim=0)  # [num_nodes, 1, vocab_size]
            all_logits = all_logits.squeeze(1)         # [num_nodes, vocab_size]
            if temp > 0:
                all_logits = all_logits / temp
            log_probs = F.log_softmax(all_logits, dim=-1)  # [num_nodes, vocab_size]
            
            next_layer_indices = []
            for i, parent_idx in enumerate(current_layer_indices):
                parent_score = score_list[parent_idx]
                
                node_log_probs = log_probs[i]  # [vocab_size]
                top_k_values, top_k_indices = torch.topk(node_log_probs, min(self.branch_n, node_log_probs.shape[-1]))
                
                for j in range(len(top_k_indices)):
                    node_idx = len(score_list)
                    token_id_list.append(top_k_indices[j].item())
                    score_list.append(parent_score + top_k_values[j].item())
                    parent_idx_list.append(parent_idx)
                    next_layer_indices.append(node_idx)
            
            if len(next_layer_indices) > self.max_nodes_per_level:
                next_layer_indices.sort(key=lambda idx: score_list[idx], reverse=True)
                next_layer_indices = next_layer_indices[:self.max_nodes_per_level]
            
            prev_selected_len = len(selected_list)
            selected_list.extend(next_layer_indices)
            depth_list.extend([depth] * len(next_layer_indices))
            
            if self.verbose:
                self.cuslog.info(f"[draft_with_tree] Layer {depth}: generated {len(next_layer_indices)} nodes")
            
           
        
        if past_kv_draft is not None:
            past_kv_draft.crop(original_kv_len)
        
        if self.max_nodes_global and len(selected_list) > self.max_nodes_global:
            node_scores = [(idx, score_list[idx], depth_list[i]) for i, idx in enumerate(selected_list)]
            node_scores.sort(key=lambda x: x[1], reverse=True)
            kept_nodes = node_scores[:self.max_nodes_global]
            selected_list = [x[0] for x in kept_nodes]
            depth_list = [x[2] for x in kept_nodes]
            
            if self.verbose:
                self.cuslog.info(f"[draft_with_tree] Global pruning: kept {len(selected_list)} nodes")
        
        if self.verbose:
            self.cuslog.info(f"[draft_with_tree] Tree generation completed: {len(selected_list)} selected nodes")
        
        depth_sorted_data = [(depth_list[i], selected_list[i], i) for i in range(len(selected_list))]
        depth_sorted_data.sort(key=lambda x: x[0])
        
        sorted_selected_list = [x[1] for x in depth_sorted_data]
        sorted_depth_list = [x[0] for x in depth_sorted_data]
        
        
        num_tokens = len(sorted_selected_list)
        node_to_position = {node_idx: pos for pos, node_idx in enumerate(sorted_selected_list)}
        
        tree_mask = torch.zeros((num_tokens, num_tokens), device=device, dtype=torch.float32)
        
        for i, node_idx in enumerate(sorted_selected_list):
            parent_node_idx = parent_idx_list[node_idx]
            
            if parent_node_idx == -1:
                tree_mask[i, i] = 1.0
            else:
                if parent_node_idx in node_to_position:
                    parent_pos = node_to_position[parent_node_idx]
                    tree_mask[i, :] = tree_mask[parent_pos, :]
                tree_mask[i, i] = 1.0
        
        attention_mask = tree_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, num_tokens, num_tokens]

        return {
            'score_list': score_list,
            'parent_idx_list': parent_idx_list,
            'token_id_list': token_id_list,
            'depth_list': sorted_depth_list,
            'selected_list': sorted_selected_list,
            'attention_mask': attention_mask,  # [1, 1, num_tokens, num_tokens]
            'updated_kv': updated_kv if accepted_tokens is not None else None
        }
    
    def build_tree_attention_mask(self, parent_indices, device):
        T = len(parent_indices)
        mask = torch.full((T, T), float('-inf'), device=device)
        
        for i in range(T):
            j = i
            while j != -1:
                mask[i, j] = 0.0
                if j < len(parent_indices):
                    j = parent_indices[j]
                else:
                    break
        
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        
        return mask
    
    def verify_tree_paths(
        self,
        tree_data: dict,
        target_logits: torch.Tensor,
        temperature: float,
    ):
        target_logits = target_logits.squeeze(0)
        device = target_logits.device
        
        selected_list = tree_data["selected_list"]
        parent_idx_list = tree_data["parent_idx_list"]
        token_id_list = tree_data["token_id_list"]
        
        T = len(selected_list)
        if T == 0:
            return None, None, None, None, -1
        
        draft_tokens = torch.tensor([token_id_list[idx] for idx in selected_list], device=device)
        
        node_to_position = {node_idx: pos for pos, node_idx in enumerate(selected_list)}
        logit_indices = []
        
        for node_idx in selected_list:
            parent_idx = parent_idx_list[node_idx]
            if parent_idx == -1:
                logit_indices.append(0)
            else:
                parent_pos = node_to_position[parent_idx]
                logit_indices.append(parent_pos + 1)
                
        logit_indices = torch.tensor(logit_indices, device=device, dtype=torch.long)
        
        verification_logits = target_logits[logit_indices]
        
        if temperature == 0:
            target_tokens = verification_logits.argmax(dim=-1)
        else:
            probs = F.softmax(verification_logits / temperature, dim=-1)
            target_tokens = torch.multinomial(probs, 1).squeeze(-1)
            
        accepted_bool = (draft_tokens == target_tokens) # [T]
        
        is_leaf = torch.ones(T, dtype=torch.bool, device=device)
        for i, node_idx in enumerate(selected_list):
            parent_idx = parent_idx_list[node_idx]
            if parent_idx != -1 and parent_idx in node_to_position:
                parent_pos = node_to_position[parent_idx]
                is_leaf[parent_pos] = False
        
        leaf_indices = torch.nonzero(is_leaf, as_tuple=True)[0].tolist()
        
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
            
        accepted_before_fly = path_accepted.clone() & path_valid_mask

        if self.enable_fly and max_len >= self.win_len:
            # self.pattern: [win_len] (False, True, True...)
            if not hasattr(self, 'pattern') or self.pattern.device != device:
                 self.pattern = torch.ones(self.win_len, dtype=torch.bool, device=device)
                 self.pattern[0] = False
            
            # Unfold path_accepted: [num_paths, num_windows, win_len]
            if max_len >= self.win_len:
                unfold_accept = path_accepted.unfold(1, self.win_len, 1) 
                matched = torch.all(unfold_accept == self.pattern, dim=-1) # [num_paths, num_windows]
                
                updated_mask = torch.zeros_like(path_accepted, dtype=torch.bool, device=device)
                updated_mask[:, :matched.shape[1]] = matched
                
                path_accepted = path_accepted | updated_mask
                
                path_accepted[:, -self.win_len:] = path_accepted[:, -self.win_len:] & accepted_before_fly[:, -self.win_len:]
                
        path_accepted = path_accepted & path_valid_mask
        cumulative_accepted = torch.cumprod(path_accepted.float(), dim=1).bool()
        final_accepted = cumulative_accepted
        
        accepted_lengths = final_accepted.sum(dim=1)
        best_path_id = accepted_lengths.argmax().item()
        best_len = accepted_lengths[best_path_id].item()
        
        if best_len == 0:
             best_draft = torch.empty((1, 0), dtype=torch.long, device=device)
             best_target = torch.empty((1, 0), dtype=torch.long, device=device)
             best_accepted_mask = torch.empty((1, 0), dtype=torch.bool, device=device)
             best_accepted_before_fly_mask = torch.empty((1, 0), dtype=torch.bool, device=device)
        else:
            best_draft = path_draft_tokens[best_path_id, :best_len].unsqueeze(0)
            best_target = path_target_tokens[best_path_id, :best_len].unsqueeze(0)
            
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
        init_len = input_ids.shape[1]
        self.reset_counter()

        input_ids = input_ids.to(self.draft_model.device)  # [1, seq_len]
        
        out_d = self.draft_model(input_ids[:, :-1], use_cache=True, return_dict=True)
        past_kv_draft = out_d.past_key_values
        out_t = self.target_model(input_ids[:, :-1], use_cache=True, return_dict=True)
        past_kv_target = out_t.past_key_values

        seed_for_next_d = input_ids[:, -1:]
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

            self.end_time = time.time()
            self.num_emitted_tokens = torch.tensor(outputs.shape[1] - init_len)
            self.num_draft_round = torch.tensor(outputs.shape[1] - init_len)
            self.num_accepted_tokens = 0

            return outputs


        draft_round = 0
        while (input_ids.shape[1] - init_len) < self.total_gen_tok:
            draft_round += 1

            if self.tree_verify:
                if self.verbose:
                    self.cuslog.info(f"[Tree Verify] Round {draft_round}: Starting tree draft generation")
                
                accepted_tokens_for_tree = None
                tree_seed = seed_for_next_d
                
                if hasattr(self, '_last_accepted_tokens') and self._last_accepted_tokens is not None:
                    if self.bonus_tok_from_target == 2:
                        tree_seed = seed_for_next_d[:, -1:]
                        accepted_tokens_for_tree = torch.cat([self._last_accepted_tokens, seed_for_next_d[:, :-1]], dim=1)
                    else:
                        tree_seed = seed_for_next_d
                        accepted_tokens_for_tree = self._last_accepted_tokens
                
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
                
                selected_list = tree_data['selected_list']
                token_id_list = tree_data['token_id_list']
                all_tree_tokens = [token_id_list[idx] for idx in selected_list]
                
                device = seed_for_next_t.device
                num_tokens = len(all_tree_tokens)
                
                flat_tokens_tensor = torch.tensor(all_tree_tokens, device=device, dtype=torch.long).unsqueeze(0)  # [1, num_tokens]
                
                seed_token = seed_for_next_t[:, -1:]  # [1, 1]
                target_in = torch.cat([seed_token, flat_tokens_tensor], dim=1)  # [1, 1+num_tokens]
                
                past_kv_len = past_kv_target.get_seq_length() if past_kv_target is not None else 0
                
                tree_mask = tree_data['attention_mask']  # [1, 1, num_tokens, num_tokens]
                
                current_len = 1 + num_tokens
                total_len = past_kv_len + current_len
                
                # current_len = 1 + num_tokens (seed + tree tokens)
                # total_len = past_kv_len + current_len
                full_attention_mask = torch.full(
                    (1, 1, current_len, total_len),
                    float('-inf'),
                    device=device,
                    dtype=self.target_model.dtype
                )
                
                # 1. Seed token (index 0 in current_len, index past_kv_len in total_len)
                full_attention_mask[:, :, 0, :past_kv_len] = 0.0
                full_attention_mask[:, :, 0, past_kv_len] = 0.0
                
                # 2. Tree tokens (indices 1..num_tokens in current_len)
                full_attention_mask[:, :, 1:, :past_kv_len] = 0.0
                full_attention_mask[:, :, 1:, past_kv_len] = 0.0
                
                tree_mask_val = torch.where(tree_mask > 0, 0.0, float('-inf')).to(self.target_model.dtype)
                full_attention_mask[:, :, 1:, past_kv_len + 1:] = tree_mask_val
                
                position_ids = torch.arange(past_kv_len, past_kv_len + current_len, device=device, dtype=torch.long).unsqueeze(0)
                
                target_outputs = self.target_model(
                    input_ids=target_in,  # [1, 1+num_tokens]
                    attention_mask=full_attention_mask,  # [1, 1, current_len, total_len]
                    position_ids=position_ids,
                    past_key_values=past_kv_target,
                    use_cache=True,
                    return_dict=True,
                )
                
                # Pass full logits (including seed output) to verify_tree_paths
                target_logits = target_outputs.logits  # [1, 1+T, vocab_size]
                
                best_draft_tokens, best_accepted, best_accepted_before_fly, best_recovered, best_path_id = self.verify_tree_paths(
                    tree_data, target_logits, temperature
                )
                
                
                if self.verbose:
                    accepted_count = best_accepted.sum().item()
                    path_len = best_accepted.shape[1]
                    self.cuslog.info(f"[Tree Verify] Best path {best_path_id}: accepted {accepted_count}/{path_len} tokens")
                
                bonus_token_ids = None
                
                if best_accepted.all():
                    bonus_logits = target_outputs.logits[:, -1:, :]  # [1, 1, vocab_size]
                    
                    if temperature == 0:
                        bonus_token_ids = bonus_logits.argmax(dim=-1)  # [1, 1]
                    else:
                        bonus_token_ids = sample_with_temperature(bonus_logits, temperature, 1)  # [1, 1]
                
                
                else:
                    first_reject_idx = (best_accepted == 0).max(1).indices.item()
                
                if self.enable_statistics:
                    initial_mismatch = (~best_accepted_before_fly).sum().item()
                    self.total_initial_mismatch += initial_mismatch
                    
                    if self.enable_fly:
                        fly_accepted = (~best_accepted_before_fly) & best_accepted
                        self.total_fly_accepted += fly_accepted.sum().item()
                
                newly_input_ids = self._create_output(
                    best_accepted, best_recovered, best_draft_tokens, bonus_token_ids, update_counter=True
                )
                
                if newly_input_ids.shape[1] == 0:
                    seed_logits = target_logits[:, 0, :]
                    
                    if temperature == 0:
                        recovery_token = seed_logits.argmax(dim=-1, keepdim=True)
                    else:
                        recovery_token = sample_with_temperature(seed_logits.unsqueeze(1), temperature, 1)
                    
                    newly_input_ids = recovery_token
                    
                    draft_token_ids = recovery_token
                    accepted = torch.ones_like(recovery_token, dtype=torch.bool)
                    
                    if self.verbose:
                        self.cuslog.info("[Tree Verify] All paths rejected. Fallback to Target Model generation.")
                else:
                    draft_token_ids = best_draft_tokens
                    accepted = best_accepted
                current_k = draft_token_ids.shape[1]
                
                if self.verbose:
                    self.cuslog.info(f"[Tree Verify] Draft tokens shape: {draft_token_ids.shape}, accepted shape: {accepted.shape}")
                
                
                
            elif self.use_ngram:
                draft_token_ids = self.draft_with_ngram(input_ids, past_kv_draft, 0)
                draft_token_ids = draft_token_ids[:,:self.k]

            else:
                draft_token_ids = []
                last = seed_for_next_d  # [1,1]
                for i in range(self.k):
                    
                    out_d = self.draft_model(input_ids=last, use_cache=True, return_dict=True, past_key_values=past_kv_draft)
                    next_token = sample_with_temperature(out_d.logits[:, -1:, :], 0)  # [1,1]
                    draft_token_ids.append(next_token)            # [1]
                    last = next_token

                draft_token_ids = torch.cat(draft_token_ids, dim=1)  # [1, k]

            if not self.tree_verify:
                seed_token = seed_for_next_t[:, -1:]  # [1, 1]
                target_in = torch.cat([seed_token, draft_token_ids], dim=1)  # [1, k+1]
                
                # Target model forward
                target_outputs = self.target_model(
                    input_ids=target_in,  # [1, k+1]
                    past_key_values=past_kv_target,
                    use_cache=True,
                    return_dict=True
                )
               
                current_k = draft_token_ids.shape[1]

                target_logits_k = target_outputs.logits[:, :-1, :][:, -current_k:, :]  # [1, k, vocab_size]

                bonus_logits = target_outputs.logits[:, -1:, :]  # [1, 1, vocab_size]
                bonus_token_ids = sample_with_temperature(bonus_logits, temperature)[0]  # [1, 1]

                accepted, recovered_token_ids = self._batch_verification(
                    target_logits_k,        # [B, k, V]
                    draft_token_ids,        # [B, k]
                    temperature,
                )

                if self.enable_statistics:
                    accepted_before_fly = accepted.clone()

                if self.entropy_thre:
                    rej_mask = self.rej_by_entropy(accepted, target_logits_k, self.entropy_thre)

                if self.enable_fly:
                    if accepted.shape[1] >= self.win_len:
                        self.pattern = torch.ones(self.win_len, dtype=torch.bool, device=accepted.device)  # [win_len]
                        self.pattern[0] = False

                        unfold_accept = accepted.unfold(1, self.win_len, 1)  # [1, num_windows, win_len]
                        matched = torch.all(unfold_accept==self.pattern, dim=-1)  # [1, num_windows]

                        updated_mask = torch.zeros_like(accepted, dtype=torch.bool, device=accepted.device)  # [1, k]
                        updated_mask[:, :matched.shape[1]] = matched
                        updated_accept = accepted | updated_mask

                        updated_accept[:, -self.win_len:] = updated_accept[:, -self.win_len:] & accepted[:, -self.win_len:]
                        
                        if self.enable_statistics:
                            fly_accepted = (~accepted_before_fly) & updated_accept
                            self.total_fly_accepted += fly_accepted.sum().item()
                        
                        accepted = updated_accept
                
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

            if not self.tree_verify:
                if accepted.all():
                    self.bonus_tok_from_target = 2
                    past_kv_draft.crop(int(input_ids.shape[1]-2))
                    seed_for_next_d = input_ids[:,-2:]
                else:
                    self.bonus_tok_from_target = 1
                    past_kv_draft.crop(int(input_ids.shape[1]-1))
                    seed_for_next_d = input_ids[:,-1:]
            else:
                newly_len = newly_input_ids.shape[1]
                if self.verbose:
                    self.cuslog.info(f"[Tree Verify] Generated {newly_len} new tokens in this round")
                
                actual_accepted = 0
                if accepted.all():
                    actual_accepted = accepted.shape[1]
                else:
                    first_false_idx = (accepted == 0).max(1).indices.item()
                    actual_accepted = first_false_idx
                
                if actual_accepted > 0 and best_draft_tokens is not None:
                    self._last_accepted_tokens = best_draft_tokens[:, :actual_accepted]
                else:
                    self._last_accepted_tokens = None
                
                if accepted.all():
                    self.bonus_tok_from_target = 2
                    seed_for_next_d = input_ids[:, -2:]
                else:
                    self.bonus_tok_from_target = 1
                    seed_for_next_d = input_ids[:, -1:]

            past_kv_target.crop(int(input_ids.shape[1] - 1))
            seed_for_next_t = input_ids[:,-1:]

            eos_found_mask = (newly_input_ids == self.tokenizer.eos_token_id) 
            eos_found_in_chunk = torch.any(eos_found_mask)
            if eos_found_in_chunk:
                break
        
        self.bonus_tok_from_target = 1
        self.end_time = time.time()


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
        
        if self.enable_statistics:
            fly_accept_rate = (self.total_fly_accepted / self.total_initial_mismatch * 100) if self.total_initial_mismatch > 0 else 0.0
            total_final_rejected = self.total_initial_mismatch - self.total_fly_accepted
            status_msg += f"|InitialMismatch:{self.total_initial_mismatch}|FLyAccepted:{self.total_fly_accepted}|FinalRejected:{total_final_rejected}|FLyAcceptRate:{fly_accept_rate:.2f}%"
            
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
        
        if self.enable_statistics:
            result["initial_mismatch"] = self.total_initial_mismatch
            result["fly_accepted"] = self.total_fly_accepted
            result["final_rejected"] = self.total_initial_mismatch - self.total_fly_accepted
            result["fly_accept_rate"] = (self.total_fly_accepted / self.total_initial_mismatch * 100) if self.total_initial_mismatch > 0 else 0.0
        
        return result
    
    def get_statistics(self):
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
        if not self.enable_statistics:
            return None
        
        global_fly_accept_rate = (self.global_total_fly_accepted / self.global_total_initial_mismatch * 100) if self.global_total_initial_mismatch > 0 else 0.0
        global_total_final_rejected = self.global_total_initial_mismatch - self.global_total_fly_accepted
        
        avg_initial_mismatch = self.global_total_initial_mismatch / self.global_sample_count if self.global_sample_count > 0 else 0.0
        avg_fly_accepted = self.global_total_fly_accepted / self.global_sample_count if self.global_sample_count > 0 else 0.0
        avg_final_rejected = global_total_final_rejected / self.global_sample_count if self.global_sample_count > 0 else 0.0
        
        return {
            "global_total_initial_mismatch": self.global_total_initial_mismatch,
            "global_total_fly_accepted": self.global_total_fly_accepted,
            "global_total_final_rejected": global_total_final_rejected,
            "global_fly_accept_rate": global_fly_accept_rate,
            "avg_initial_mismatch": avg_initial_mismatch,
            "avg_fly_accepted": avg_fly_accepted,
            "avg_final_rejected": avg_final_rejected,
            "avg_fly_accept_rate": global_fly_accept_rate,
            "sample_count": self.global_sample_count,
        }
    
    def print_global_statistics(self):
        if not self.enable_statistics:
            return
        
        stats = self.get_global_statistics()
        if stats is None:
            return
        
        self.cuslog.info("=" * 80)
        self.cuslog.info("Global Statistics")
        self.cuslog.info("=" * 80)
        self.cuslog.info(f"Total Samples: {stats['sample_count']}")
        self.cuslog.info("")
        self.cuslog.info("Totals:")
        self.cuslog.info(f"  Initial Mismatch: {stats['global_total_initial_mismatch']}")
        self.cuslog.info(f"  FLy Accepted: {stats['global_total_fly_accepted']}")
        self.cuslog.info(f"  Final Rejected: {stats['global_total_final_rejected']}")
        self.cuslog.info(f"  FLy Accept Rate: {stats['global_fly_accept_rate']:.2f}%")
        self.cuslog.info("")
        self.cuslog.info("Averages per Sample:")
        self.cuslog.info(f"  Avg Initial Mismatch: {stats['avg_initial_mismatch']:.2f}")
        self.cuslog.info(f"  Avg FLy Accepted: {stats['avg_fly_accepted']:.2f}")
        self.cuslog.info(f"  Avg Final Rejected: {stats['avg_final_rejected']:.2f}")
        self.cuslog.info(f"  Avg FLy Accept Rate: {stats['avg_fly_accept_rate']:.2f}%")
        self.cuslog.info("=" * 80)
    
    def reset_counter(self):
        self._counter_inited = False
        if self.enable_statistics:
            self.total_initial_mismatch = 0
            self.total_fly_accepted = 0
        if self.tree_verify:
            self._last_accepted_tokens = None
    
    def decode_ids(self, ids):
        token = self.tokenizer.decode(ids[0], skip_special_tokens=False)
        return token
