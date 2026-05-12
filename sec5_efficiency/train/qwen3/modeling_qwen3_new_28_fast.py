from .modeling_qwen3_new_28 import *  # noqa: F401,F403
from .modeling_qwen3_new_28 import Qwen3ForCausalLM as _BaseQwen3ForCausalLM


class Qwen3ForCausalLM(_BaseQwen3ForCausalLM):
    @torch.no_grad()
    def medusa_generate_interleaved_multihead_stream_user_same_y0(
        self,
        question_text: str,
        assistant_heads: int = 4,
        assistant_prefix_texts=None,
        assistant_prefill_texts=None,
        max_new_tokens: int = 1024,
        max_steps: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 0,
        do_sample: bool = False,
        stop_on_im_end: bool = True,
        allow_same_step_visible: bool = False,
        presence_penalty: float = 0.0,
    ) -> Dict[int, torch.Tensor]:
        """
        Faster equivalent implementation of the original multi-head streaming path.

        Optimizations kept strictly at the implementation level:
        - reuse per-step buffers instead of allocating rows/pos/input tensors each step
        - avoid string-based active_kinds bookkeeping
        - batch lm_head on assistant rows for init and decode
        - use in-place masked_fill_ instead of torch.where + zeros_like

        Attention semantics, stop semantics, and output format are preserved.
        """

        device = next(self.parameters()).device
        tokenizer = self.tokenizer
        H = int(assistant_heads)
        assert H >= 1
        last_head = H - 1

        def _norm_texts(x, default=""):
            if x is None:
                return [default] * H
            if isinstance(x, str):
                return [x] * H
            xs = list(x)
            if len(xs) == 1 and H > 1:
                return xs * H
            assert len(xs) == H, f"Expect len={H}, got len={len(xs)}"
            return xs

        assistant_prefix_texts = _norm_texts(assistant_prefix_texts, default="")
        assistant_prefill_texts = _norm_texts(assistant_prefill_texts, default="")

        im_start = int(self.im_start)
        im_end = int(self.im_end)
        nl = int(self.newline_token)

        sys_tokens = list(self.system_tokens)
        user_tokens = list(self.user_tokens)
        asst_tokens = list(self.assistant_tokens)

        system_prefix = [im_start] + sys_tokens + [nl]
        user_prefix = [im_start] + user_tokens + [nl]
        asst_prefix = [im_start] + asst_tokens + [nl]

        sys_msg = getattr(self, "system_message", "You are a helpful assistant.")
        sys_ids = tokenizer.encode(sys_msg, add_special_tokens=False)
        user_content_ids = tokenizer.encode(question_text, add_special_tokens=False)
        user_stream = user_content_ids + [im_end]

        head_prefix_ids: List[List[int]] = []
        head_prefill_ids: List[List[int]] = []
        for h in range(H):
            ptxt = assistant_prefix_texts[h]
            ctxt = assistant_prefill_texts[h]
            head_prefix_ids.append(tokenizer.encode(ptxt, add_special_tokens=False) if ptxt else [])
            head_prefill_ids.append(tokenizer.encode(ctxt, add_special_tokens=False) if ctxt else [])

        step0_ids: List[int] = []
        step0_ids += system_prefix + sys_ids + [im_end]
        step0_ids += user_prefix

        head_block_ranges: List[Tuple[int, int]] = []
        for h in range(H):
            bs = len(step0_ids)
            step0_ids += asst_prefix + head_prefix_ids[h] + head_prefill_ids[h]
            be = len(step0_ids)
            head_block_ranges.append((bs, be))

        input_ids0 = torch.tensor([step0_ids], device=device, dtype=torch.long)
        S0 = input_ids0.size(1)

        sys_len = len(system_prefix) + len(sys_ids) + 1
        y0 = sys_len
        user_prefix_len = len(user_prefix)
        user_bs = sys_len
        user_pe = sys_len + user_prefix_len

        asst_prefix_len = len(asst_prefix)
        head_prefix_len = [len(x) for x in head_prefix_ids]
        head_prefill_len = [len(x) for x in head_prefill_ids]

        pos0 = torch.zeros((S0, 2), device=device, dtype=torch.long)
        pos0[:sys_len, 0] = 0
        pos0[:sys_len, 1] = torch.arange(sys_len, device=device, dtype=torch.long)
        pos0[user_bs:user_pe, 0] = 1
        pos0[user_bs:user_pe, 1] = y0 + torch.arange(user_prefix_len, device=device, dtype=torch.long)
        for h in range(H):
            owner = 2 + h
            bs, be = head_block_ranges[h]
            block_len = be - bs
            if block_len <= 0:
                continue
            pos0[bs:be, 0] = owner
            pos0[bs:be, 1] = y0 + torch.arange(block_len, device=device, dtype=torch.long)

        token_y0 = pos0[:, 1].clone()

        def _minf(dtype: torch.dtype) -> float:
            return -1e9 if dtype == torch.float32 else -1e4

        mask_dtype = next(self.parameters()).dtype
        minf = _minf(mask_dtype)
        attn0 = torch.full((S0, S0), minf, device=device, dtype=mask_dtype)
        idx0 = torch.arange(S0, device=device)
        row_idx0 = idx0.unsqueeze(1)
        col_idx0 = idx0.unsqueeze(0)
        attn0.masked_fill_(token_y0.unsqueeze(0) < token_y0.unsqueeze(1), 0.0)
        attn0.masked_fill_((row_idx0 < sys_len) & (col_idx0 <= row_idx0), 0.0)
        attn0.masked_fill_((row_idx0 >= user_bs) & (row_idx0 < user_pe) & (col_idx0 >= user_bs) & (col_idx0 <= row_idx0), 0.0)
        for h in range(H):
            bs, be = head_block_ranges[h]
            if be <= bs:
                continue
            attn0.masked_fill_((row_idx0 >= bs) & (row_idx0 < be) & (col_idx0 >= bs) & (col_idx0 <= row_idx0), 0.0)
        attn0[idx0, idx0] = 0.0

        out0 = self.model(
            input_ids=input_ids0,
            attention_mask=attn0.unsqueeze(0).unsqueeze(0),
            position_ids=pos0.unsqueeze(0),
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        past_kv = out0.past_key_values

        max_cache_len = S0 + len(user_stream) + H * max_new_tokens
        cache_len = S0

        num_owners = H + 2
        owner_pos_buf = torch.empty((num_owners, max_cache_len), device=device, dtype=torch.long)
        owner_y_buf = torch.empty((num_owners, max_cache_len), device=device, dtype=torch.long)
        owner_lens = [0 for _ in range(num_owners)]
        owner_lens_t = torch.zeros((num_owners,), device=device, dtype=torch.long)
        owner_ids0 = pos0[:, 0]
        for idx in range(S0):
            owner = int(owner_ids0[idx].item())
            off = owner_lens[owner]
            owner_pos_buf[owner, off] = idx
            owner_y_buf[owner, off] = token_y0[idx]
            owner_lens[owner] = off + 1
            owner_lens_t[owner] = off + 1

        generated_prefix = step0_ids[:]
        generated_tail: List[int] = []
        gen_len = len(generated_prefix)

        user_ptr = 0
        assistant_histories: List[List[int]] = [[] for _ in range(H)]
        head_token_positions: List[List[int]] = [[] for _ in range(H)]
        pending_asst: List[Optional[int]] = [None for _ in range(H)]
        asst_gen_fed: List[int] = [0 for _ in range(H)]
        asst_gen_count: List[int] = [0 for _ in range(H)]
        global_stop = False

        local_prefix_total = [asst_prefix_len + head_prefix_len[h] + head_prefill_len[h] for h in range(H)]

        qmax = 1 + H
        diag_idx = torch.arange(qmax, device=device)
        head_ids_buf = torch.empty((qmax,), device=device, dtype=torch.long)
        tri_prev = None
        if allow_same_step_visible and qmax > 1:
            tri_prev = torch.tril(torch.ones((qmax, qmax), device=device, dtype=torch.bool), diagonal=-1)

        feed_buf = torch.empty((1, qmax), device=device, dtype=torch.long)
        pos_q_buf = torch.empty((1, qmax, 2), device=device, dtype=torch.long)
        rows_buf = torch.empty((qmax, max_cache_len + qmax), device=device, dtype=mask_dtype)
        assistant_row_idx_buf = torch.empty((H,), device=device, dtype=torch.long)
        prefix_idx_buf = torch.arange(max_cache_len, device=device, dtype=torch.long).unsqueeze(0)

        ctx_positions = torch.empty((H,), device=device, dtype=torch.long)
        for h in range(H):
            bs, be = head_block_ranges[h]
            ctx_positions[h] = (be - 1) if be > bs else (S0 - 1)
        init_hidden = out0.last_hidden_state[0, ctx_positions]
        init_logits = self.lm_head(init_hidden)

        for h in range(H):
            first_tok = int(
                self._select_next_token(
                    logits_1d=init_logits[h],
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=0.0,
                    presence_penalty=presence_penalty,
                    past_tokens=assistant_histories[h],
                )
            )
            assistant_histories[h].append(first_tok)
            generated_tail.append(first_tok)
            head_token_positions[h].append(gen_len)
            gen_len += 1
            asst_gen_count[h] += 1
            if stop_on_im_end and first_tok == im_end:
                pending_asst[h] = None
                if h == last_head:
                    global_stop = True
            else:
                pending_asst[h] = first_tok

        if global_stop:
            for hh in range(H):
                pending_asst[hh] = None

        for _ in range(max_steps):
            if global_stop:
                break
            if user_ptr >= len(user_stream):
                if all((p is None or asst_gen_count[h] >= max_new_tokens) for h, p in enumerate(pending_asst)):
                    break

            q = 0
            has_user = False

            if user_ptr < len(user_stream):
                has_user = True
                feed_buf[0, q] = int(user_stream[user_ptr])
                pos_q_buf[0, q, 0] = 1
                pos_q_buf[0, q, 1] = max(0, min(int(y0 + user_prefix_len + user_ptr), int(self.max_position_embeddings) - 1))
                head_ids_buf[q] = -1
                q += 1

            for h in range(H):
                if pending_asst[h] is None:
                    continue
                if asst_gen_count[h] >= max_new_tokens:
                    pending_asst[h] = None
                    continue
                feed_buf[0, q] = int(pending_asst[h])
                pos_q_buf[0, q, 0] = 2 + h
                pos_q_buf[0, q, 1] = max(
                    0,
                    min(
                        int(y0 + local_prefix_total[h] + asst_gen_fed[h]),
                        int(self.max_position_embeddings) - 1,
                    ),
                )
                head_ids_buf[q] = h
                q += 1

            if q == 0:
                break

            key_len = cache_len + q
            inp = feed_buf[:, :q]
            pos_q = pos_q_buf[:, :q, :]
            yq_t = pos_q[0, :, 1]

            rows = rows_buf[:q, :key_len]
            rows.fill_(minf)
            for owner in range(num_owners):
                owner_len = int(owner_lens_t[owner].item())
                if owner_len <= 0:
                    continue
                owner_y = owner_y_buf[owner, :owner_len]
                vis_lens = torch.searchsorted(owner_y, yq_t, right=False)
                max_vis = int(vis_lens.max().item())
                if max_vis <= 0:
                    continue
                owner_pos = owner_pos_buf[owner, :max_vis]
                prefix_mask = prefix_idx_buf[:, :max_vis] < vis_lens.unsqueeze(1)
                rows[:, owner_pos].masked_fill_(prefix_mask, 0.0)
            di = diag_idx[:q]
            rows[di, cache_len + di] = 0.0
            if allow_same_step_visible and q > 1:
                rows[:, cache_len : cache_len + q].masked_fill_(tri_prev[:q, :q], 0.0)

            out = self.model(
                input_ids=inp,
                attention_mask=rows.unsqueeze(0).unsqueeze(0),
                position_ids=pos_q,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values

            if cache_len + q > max_cache_len:
                raise RuntimeError(f"cache buffer overflow: cache_len={cache_len}, Q={q}, max_cache_len={max_cache_len}")
            for i in range(q):
                owner = int(pos_q[0, i, 0].item())
                off = owner_lens[owner]
                owner_pos_buf[owner, off] = cache_len + i
                owner_y_buf[owner, off] = yq_t[i]
                owner_lens[owner] = off + 1
                owner_lens_t[owner] = off + 1
            cache_len += q

            stop_triggered_this_step = False
            if has_user:
                user_ptr += 1

            num_asst_rows = 0
            for i in range(q):
                h = int(head_ids_buf[i].item())
                if h < 0:
                    continue
                assistant_row_idx_buf[num_asst_rows] = i
                num_asst_rows += 1
                asst_gen_fed[h] += 1

            if num_asst_rows:
                row_idx_t = assistant_row_idx_buf[:num_asst_rows]
                hidden_batch = out.last_hidden_state[0, row_idx_t]
                logits_batch = self.lm_head(hidden_batch)
                for j in range(num_asst_rows):
                    h = int(head_ids_buf[int(row_idx_t[j].item())].item())
                    tok_next = int(
                        self._select_next_token(
                            logits_1d=logits_batch[j],
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            min_p=0.0,
                            presence_penalty=presence_penalty,
                            past_tokens=assistant_histories[h],
                        )
                    )
                    assistant_histories[h].append(tok_next)
                    generated_tail.append(tok_next)
                    head_token_positions[h].append(gen_len)
                    gen_len += 1
                    asst_gen_count[h] += 1

                    if stop_on_im_end and tok_next == im_end:
                        pending_asst[h] = None
                        if h == last_head:
                            stop_triggered_this_step = True
                    elif asst_gen_count[h] >= max_new_tokens:
                        pending_asst[h] = None
                    else:
                        pending_asst[h] = tok_next

            if stop_triggered_this_step:
                global_stop = True
                for hh in range(H):
                    pending_asst[hh] = None
                break

        all_ids = torch.tensor([generated_prefix + generated_tail], device=device, dtype=torch.long)
        return self._extract_outputs(
            head_token_positions=head_token_positions,
            generated_ids=all_ids,
            stop_at_im_end=True,
            stop_at_eos=False,
        )
