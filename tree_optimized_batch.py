# 优化的批量处理版本（可选）
# 这是一个更高效的批量处理实现，可以作为参考

def draft_with_tree_optimized(self, seed_token, past_kv_draft, temp=0):
    """
    树形草稿生成：批量处理优化版本
    
    通过对相同深度的节点进行批量处理来提高效率。
    使用 padding 来处理不同长度的路径。
    """
    device = seed_token.device
    original_kv_len = past_kv_draft.get_seq_length() if past_kv_draft is not None else 0
    
    # ... 前面的初始化代码保持不变 ...
    
    # 逐层扩展树（优化的批量处理）
    for depth in range(2, self.k + 1):
        if len(current_layer_indices) == 0:
            break
        
        # 收集所有路径并进行 padding
        max_path_len = depth  # 当前深度的最大路径长度
        batch_paths = []
        path_lengths = []
        
        for parent_idx in current_layer_indices:
            # 构建从 root 到当前节点的路径
            path_tokens = []
            current_idx = parent_idx
            while current_idx != -1:
                path_tokens.append(all_nodes[current_idx].token_id)
                current_idx = all_nodes[current_idx].parent_idx
            path_tokens.reverse()
            
            batch_paths.append(path_tokens)
            path_lengths.append(len(path_tokens))
        
        # Padding 到相同长度
        padded_paths = torch.zeros(len(batch_paths), max_path_len, dtype=torch.long, device=device)
        attention_mask = torch.zeros(len(batch_paths), max_path_len, dtype=torch.bool, device=device)
        
        for i, path in enumerate(batch_paths):
            padded_paths[i, :len(path)] = torch.tensor(path, device=device)
            attention_mask[i, :len(path)] = True
        
        # 批量 forward
        with torch.no_grad():
            out = self.draft_model(
                input_ids=padded_paths,  # [batch_size, max_path_len]
                attention_mask=attention_mask,  # [batch_size, max_path_len]
                past_key_values=past_kv_draft,
                use_cache=True,
                return_dict=True,
            )
        
        # 提取每个路径最后一个有效位置的 logits
        batch_logits = []
        for i, length in enumerate(path_lengths):
            last_logits = out.logits[i, length-1, :]  # [vocab_size]
            batch_logits.append(last_logits)
        
        # 立即恢复 KV cache
        past_kv_draft.crop(original_kv_len)
        
        # ... 后续的子节点生成代码保持不变 ...
        
    return tree_data


# 另一种优化策略：使用缓存避免重复计算
class TreeNodeWithCache:
    """带缓存的树节点，避免重复计算路径"""
    def __init__(self, token_id, log_prob, score, parent_idx, depth):
        self.token_id = token_id
        self.log_prob = log_prob
        self.score = score
        self.parent_idx = parent_idx
        self.depth = depth
        self.children = []
        self._path_cache = None  # 缓存从 root 到当前节点的路径
    
    def get_path(self, all_nodes):
        """获取从 root 到当前节点的路径（带缓存）"""
        if self._path_cache is None:
            path = []
            current_idx = self
            while current_idx.parent_idx != -1:
                path.append(current_idx.token_id)
                current_idx = all_nodes[current_idx.parent_idx]
            path.append(current_idx.token_id)
            path.reverse()
            self._path_cache = path
        return self._path_cache

