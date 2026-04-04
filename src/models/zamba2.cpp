#include "models.h"

llm_build_zamba2::llm_build_zamba2(const llama_model & model, const llm_graph_params & params) :
    llm_build_mamba_base(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // Embed tokens
    inpL = build_inp_embd(model.tok_embd);

    // Store original embeddings for concat at hybrid layers
    // Cast to F32 to avoid type mismatch with evolved hidden states
    ggml_tensor * inpOrig = ggml_cast(ctx0, inpL, GGML_TYPE_F32);
    cb(inpOrig, "inpOrig", -1);

    // Position embeddings for RoPE (skip if n_rot=0, i.e. use_mem_rope=false)
    ggml_tensor * inp_pos = (n_rot > 0) ? build_inp_pos() : nullptr;

    // Build hybrid memory (KV cache + recurrent state)
    auto * inp = build_inp_mem_hybrid();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_head_kv_il = hparams.n_head_kv(il);
        const bool is_hybrid = (n_head_kv_il > 0);

        ggml_tensor * residual = inpL;

        if (is_hybrid) {
            // ============ HYBRID LAYER ============
            // Step 1: Shared transformer (attention + FFN)
            // Concat current hidden states with original embeddings
            // cur: [n_embd, n_tokens], inpOrig: [n_embd, n_tokens]
            // result: [2*n_embd, n_tokens]
            cur = ggml_concat(ctx0, inpL, inpOrig, 0);
            cb(cur, "attn_concat", il);

            // Pre-attention norm (operates on 4096-dim concat)
            cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm_concat", il);

            // Self-attention on 4096-dim input
            const int64_t n_head_il = hparams.n_head(il);
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head_il, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv_il, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv_il, n_tokens);

            // RoPE (conditional: Zamba2 models with use_mem_rope=false have n_rot=0)
            if (n_rot > 0) {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                    n_rot, hparams.rope_type, n_ctx_orig,
                    freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                    n_rot, hparams.rope_type, n_ctx_orig,
                    freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // Build attention (O projects from n_heads*head_dim to n_embd)
            cur = build_attn(inp->get_attn(),
                    model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);

            // Step 2: FFN (operates on n_embd-dim output from attention)
            cur = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL, LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            // Step 3: Linear mixing (transformer output projection)
            cur = build_lora_mm(model.layers[il].ssm_mix, cur);
            cb(cur, "ssm_mix", il);

            // Step 4: Mamba-2 with transformer residual
            // mamba input = hidden_states + transformer_output
            ggml_tensor * mamba_input = ggml_add(ctx0, inpL, cur);
            cb(mamba_input, "mamba_input", il);

            // Pre-mamba norm
            cur = build_norm(mamba_input, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "mamba_norm", il);

            // Mamba-2 layer
            cur = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
            cb(cur, "mamba_out", il);

            // Residual: original hidden + mamba output
            if (il == n_layer - 1 && inp_out_ids) {
                cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
                residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            }

            cur = ggml_add(ctx0, residual, cur);

        } else {
            // ============ PURE MAMBA LAYER ============
            // Pre-mamba norm
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // Mamba-2 layer
            cur = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
            cb(cur, "mamba_out", il);

            // Residual
            if (il == n_layer - 1 && inp_out_ids) {
                cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
                residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            }

            cur = ggml_add(ctx0, residual, cur);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }

    // Final norm
    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
