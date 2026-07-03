// Standalone CPU validation of sd_teamwork_lora_delta against a plain C++ reference.
// Build with the standalone recipe in TEAMWORK_PLAN.md §3.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "model/adapter/teamwork_adapter.hpp"

int main() {
    const int in = 5, r = 3, out = 6, T = 3, L = 4, n_txt = 2, N = 2;
    const int ntok = n_txt + T * L;
    const int text_teammate = 1;
    const bool comm = true;

    struct ggml_init_params p = {(size_t)512 * 1024 * 1024, nullptr, false};
    ggml_context* ctx = ggml_init(p);

    ggml_tensor* x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in, ntok, N);
    ggml_tensor* down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, r, in, T);   // ne=[r,in,T]
    ggml_tensor* up   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, out, r, T);  // ne=[out,r,T]

    auto frand = []() { return (float)(rand() % 2000 - 1000) / 500.0f; };
    srand(1234);
    float* xd = (float*)x->data;
    for (int i = 0; i < in * ntok * N; ++i) xd[i] = frand();
    float* dd = (float*)down->data;
    for (int i = 0; i < r * in * T; ++i) dd[i] = frand();
    float* ud = (float*)up->data;
    for (int i = 0; i < out * r * T; ++i) ud[i] = frand();

    ggml_tensor* delta = sd_teamwork_lora_delta(ctx, x, down, up, n_txt, T, text_teammate, comm);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, delta);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    // reference
    auto X    = [&](int i, int tok, int n) { return xd[i + tok * in + n * in * ntok]; };
    auto DOWN = [&](int t, int i, int rho) { return dd[rho + i * r + t * r * in]; };   // down_logical[t,i,rho]
    auto UP   = [&](int t, int rho, int o) { return ud[o + rho * out + t * out * r]; }; // up_logical[t,rho,o]
    float* od = (float*)delta->data;
    auto OUT  = [&](int o, int tok, int n) { return od[o + tok * out + n * out * ntok]; };

    double max_err = 0.0;
    for (int n = 0; n < N; ++n) {
        // text tokens
        for (int tok = 0; tok < n_txt; ++tok) {
            std::vector<double> h(r, 0.0);
            for (int rho = 0; rho < r; ++rho)
                for (int i = 0; i < in; ++i) h[rho] += X(i, tok, n) * DOWN(text_teammate, i, rho);
            for (int o = 0; o < out; ++o) {
                double v = 0.0;
                for (int rho = 0; rho < r; ++rho) v += h[rho] * UP(text_teammate, rho, o);
                max_err = std::max(max_err, std::fabs(v - OUT(o, tok, n)));
            }
        }
        // image blocks
        for (int l = 0; l < L; ++l) {
            // per-teammate down, then shared hidden
            std::vector<std::vector<double>> lx(T, std::vector<double>(r, 0.0));
            std::vector<double> shared(r, 0.0);
            for (int t = 0; t < T; ++t) {
                int tok = n_txt + t * L + l;
                for (int rho = 0; rho < r; ++rho) {
                    for (int i = 0; i < in; ++i) lx[t][rho] += X(i, tok, n) * DOWN(t, i, rho);
                    shared[rho] += lx[t][rho];
                }
            }
            for (int t = 0; t < T; ++t) {
                int tok = n_txt + t * L + l;
                const std::vector<double>& h = comm ? shared : lx[t];
                for (int o = 0; o < out; ++o) {
                    double v = 0.0;
                    for (int rho = 0; rho < r; ++rho) v += h[rho] * UP(t, rho, o);
                    max_err = std::max(max_err, std::fabs(v - OUT(o, tok, n)));
                }
            }
        }
    }

    bool ok = delta->ne[0] == out && delta->ne[1] == ntok && delta->ne[2] == N && max_err < 1e-4;
    printf("delta shape = [%lld, %lld, %lld] (want [%d, %d, %d])\n",
           (long long)delta->ne[0], (long long)delta->ne[1], (long long)delta->ne[2], out, ntok, N);
    printf("max abs err vs reference = %.3e\n", max_err);
    printf("%s\n", ok ? "PASS" : "FAIL");
    fflush(stdout);
    ggml_free(ctx);
    return ok ? 0 : 1;
}
