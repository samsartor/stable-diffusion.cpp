#ifndef __SD_MODEL_ADAPTER_TEAMWORK_ADAPTER_HPP__
#define __SD_MODEL_ADAPTER_TEAMWORK_ADAPTER_HPP__

#include <vector>

#include "ggml.h"
#include "model/adapter/teamwork.hpp"

// Teamwork communicating-LoRA delta for ONE projection, in sd.cpp's sequence-axis
// layout. Returns the residual to add to the base linear output.
//
//   x:          [in,  ntok, N]   ntok = n_txt + T*L  (n_txt=0 for image-only streams)
//   down_stack: [r,   in,   T]   (safetensors [T, in, r]  -> ggml ne=[r,in,T])
//   up_stack:   [out, r,    T]   (safetensors [T, r, out] -> ggml ne=[out,r,T])
//   returns:    [out, ntok, N]
//
// Image tokens are T contiguous equal blocks (teammate-index order); block t uses
// teammate-slice t of the stacks. With communication, the down-projections of all
// teammates at the same spatial position are summed into one shared rank-r hidden,
// then each teammate reads it back through its own up-projection. Text tokens (if
// n_txt>0) use a single teammate slice (`text_teammate`) with no cross-teammate
// mixing. See TEAMWORK_PLAN.md §4.
//
// NOTE (scope): this hard-codes the ViT "[text | T equal image blocks]" layout, which
// is all the current FLUX2/Klein checkpoints need. The fully general per-token
// teammate/position-id form (arbitrary token->teammate assignment + scatter/gather
// communication) is intentionally NOT implemented: there is no motivating case yet,
// scatter/gather is slow, YAGNI. Revisit if irregular teammates / interleaved
// modalities appear (see TEAMWORK_PLAN.md §4 "general form").
//
// NOTE (perf): the T down/up matmuls are looped for clarity. They can be fused into
// batched matmuls over the teammate axis (reshape image span to [in, L, T, N] and use
// ggml_mul_mat's ne2 batching); the communicating down+sum can even collapse to a
// single matmul by concatenating teammate features (D_cat [in*T, r]). Benefit scales
// with T (modest at T=3, meaningful for large-T tasks like SVBRDF); costs extra
// reshapes/permutes, so benchmark before adopting. YAGNI for now.
static inline ggml_tensor* sd_teamwork_lora_delta(ggml_context* ctx,
                                                  ggml_tensor* x,
                                                  ggml_tensor* down_stack,
                                                  ggml_tensor* up_stack,
                                                  int n_txt,
                                                  int T,
                                                  int text_teammate,
                                                  bool communication) {
    const int64_t in   = x->ne[0];
    const int64_t ntok = x->ne[1];
    const int64_t N    = x->ne[2];
    const int64_t r    = down_stack->ne[0];
    const int64_t out  = up_stack->ne[0];
    const int64_t L    = (ntok - (int64_t)n_txt) / T;

    // Orient per-teammate weights for mul_mat (which contracts over ne[0]).
    // Dt: [in, r, T]  Ut: [r, out, T]
    ggml_tensor* Dt = ggml_cont(ctx, ggml_transpose(ctx, down_stack));
    ggml_tensor* Ut = ggml_cont(ctx, ggml_transpose(ctx, up_stack));

    auto down_slice = [&](int t) {
        return ggml_cont(ctx, ggml_view_2d(ctx, Dt, in, r, Dt->nb[1], (size_t)t * Dt->nb[2]));
    };
    auto up_slice = [&](int t) {
        return ggml_cont(ctx, ggml_view_2d(ctx, Ut, r, out, Ut->nb[1], (size_t)t * Ut->nb[2]));
    };

    // Per-teammate down projection of each contiguous image block: lx[t] = [r, L, N].
    std::vector<ggml_tensor*> lx(T);
    for (int t = 0; t < T; ++t) {
        ggml_tensor* xt = ggml_cont(ctx, ggml_view_3d(ctx, x, in, L, N,
                                                      x->nb[1], x->nb[2],
                                                      (size_t)(n_txt + (int64_t)t * L) * x->nb[1]));
        lx[t] = ggml_mul_mat(ctx, down_slice(t), xt);  // [r, L, N]
    }

    // Communication: shared low-rank hidden = sum of teammate down-projections.
    ggml_tensor* shared = nullptr;
    if (communication) {
        for (int t = 0; t < T; ++t) {
            shared = shared ? ggml_add(ctx, shared, lx[t]) : lx[t];
        }
    }

    // Per-teammate up projection, re-concatenated in teammate-index (block) order.
    ggml_tensor* img_delta = nullptr;
    for (int t = 0; t < T; ++t) {
        ggml_tensor* h  = communication ? shared : lx[t];
        ggml_tensor* ot = ggml_mul_mat(ctx, up_slice(t), h);  // [out, L, N]
        img_delta       = img_delta ? ggml_concat(ctx, img_delta, ot, 1) : ot;
    }

    if (n_txt == 0) {
        return img_delta;
    }

    // Text tokens: single teammate slice, no cross-teammate communication.
    ggml_tensor* x_txt = ggml_cont(ctx, ggml_view_3d(ctx, x, in, n_txt, N, x->nb[1], x->nb[2], 0));
    ggml_tensor* h_txt = ggml_mul_mat(ctx, down_slice(text_teammate), x_txt);
    ggml_tensor* txt_delta = ggml_mul_mat(ctx, up_slice(text_teammate), h_txt);  // [out, n_txt, N]

    return ggml_concat(ctx, txt_delta, img_delta, 1);  // [out, ntok, N]
}

#endif  // __SD_MODEL_ADAPTER_TEAMWORK_ADAPTER_HPP__
