# Teamwork support in stable-diffusion.cpp — implementation plan & knowledge base

Living checkpoint doc for porting the **Teamwork** diffusion method (token-aligned
communicating LoRA + joint attention) into this repo. Written to survive context
resets / non-persistent GPU instances. Author of the method + this effort: Sam
Sartor (ssartor@adobe.com) — *"Teamwork: Collaborative Diffusion with Low-rank
Coordination and Adaptation"*, SIGGRAPH Asia 2025.

---

## 1. Goal & scope

Add Teamwork to sd.cpp, including loading checkpoints produced by the Python/diffusers
impl. **First target: FLUX.2-Klein-4B, profile `FLUX2_PLUSATTN`**, via the "overpainting"
fine-tune (mask-guided local image editing). First consumer: a low-latency C++ backend
for `overpainteditor`.

Teammates are placed on the **sequence axis** (not the batch axis as in Python, not the
channel axis). ViTs only.

### Why Klein is the *easy* first target
sd.cpp's existing FLUX2-Klein reference-image editing already provides almost everything
Teamwork needs; the only genuinely new machinery is the communicating LoRA:

| Teamwork needs | sd.cpp already does |
|---|---|
| Teammates concatenated on the **sequence axis** | `ref_latents` concat on `ne[1]` (`flux.hpp` `forward_flux_chroma` ~1317) — target img first, then refs |
| Per-teammate **RoPE offsets 0,10,20** | Klein `RefIndexMode::INCREASE × ref_index_scale=10` (`rope.hpp` `gen_refs_ids` ~354) → exactly 0,10,20 |
| **Clean inputs pinned, output denoised** | refs are clean conditioning; target is the denoised token block — inherent to editing |
| **Joint attention** across teammates | With `attn_allow=None` + all teammates present → plain all-to-all attention over `[text|edited|source|mask]`, which Klein multi-ref attention already computes |
| Roster order ↔ token order | `edited.out`(target,id0) → `source.in`(ref0,id10) → `mask.in`(ref1,id20) matches sd.cpp target-then-refs 1:1 |

So for this checkpoint: RoPE offsets, input pinning, and attention masking are **free**.
The only new work is the per-teammate communicating LoRA.

---

## 2. Files, weights, checkpoints (all local on the GPU box)

- **Target teamwork checkpoint:** `~/sensei/checkpoints/overpainting_klein9b_jul1.safetensors`
  (478 MB). **Filename says "9b" but it is really the 4B base** (`base_model=black-forest-labs/FLUX.2-klein-4B`
  in metadata). Trust `teamwork.*`/`base_model` metadata, never the filename. `_02` variant same.
- **Base transformer:** `~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/*/flux-2-klein-4b.safetensors`
  (7.75 GB, BFL/original naming) **or** `/mnt/localssd/flux-2-klein-4b-Q8_0.gguf` (4.3 GB).
- **VAE:** same snapshot `vae/diffusion_pytorch_model.safetensors` (168 MB); load with `--vae-format flux2`.
- **Text encoder:** Qwen3-4B. sd.cpp `--llm` cannot load the sharded diffusers dir (it wrongly
  triggers the `unet/` loader). FIX ALREADY DONE: merged shards → **`/mnt/localssd/qwen3_4b_te.safetensors`**
  (8 GB; names preserved as `model.*`, `--llm` prepends `text_encoders.llm.`).
- **Python impl:** `/mnt/localssd/teamwork` (installed in `/mnt/localssd/overpainting/.venv`); key files
  `teamwork/{linear.py(LoRA math), batch.py(Selection/BatchBuilder), attn.py(joint attn),
  pipeline_flux2_klein.py, config.py, adapter.py}`.
- **Overpainting:** `/mnt/localssd/overpainting` (config `overpaintmodel.py:334`, editor `overpainteditor/main.py`
  backend Protocol ~1147, eval `evalmethods.py`).

### Verified base-Klein command (substrate works: text2img + edit)
```
./build/bin/sd-cli --diffusion-model /mnt/localssd/flux-2-klein-4b-Q8_0.gguf \
  --vae <snap>/vae/diffusion_pytorch_model.safetensors --vae-format flux2 \
  --llm /mnt/localssd/qwen3_4b_te.safetensors \
  [-r ref.png] -p "..." --cfg-scale 1.0 --steps 4 --sampling-method euler \
  -o out.png --diffusion-fa --offload-to-cpu
```

---

## 3. Build & iterate

- Env: 2× H100 80GB, CUDA 12.4, cmake 3.22, gcc 11.4.
- Configure (once): `git submodule update --init --recursive` then
  `cmake -S . -B build -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DSD_WEBP=OFF -DSD_WEBM=OFF -DCMAKE_BUILD_TYPE=Release`
- Incremental build: `cmake --build build -j$(nproc)` (only changed .cpp recompiles; fast).
- **Standalone test-link recipe** (unit-test a header against the built lib, no CMake target):
```
g++ -std=c++17 -Isrc -Iinclude -Iggml/include -Ithirdparty test.cpp \
  build/libstable-diffusion.a build/ggml/src/libggml-cpu.a build/ggml/src/ggml-cuda/libggml-cuda.a \
  build/ggml/src/libggml.a build/ggml/src/libggml-base.a \
  -lcudart -lcublas -lcuda -L/usr/local/cuda/lib64 -lpthread -ldl -lgomp -o test
```

---

## 4. Core algorithm (VALIDATED against golden dump, ~1% bf16 error)

Per adapted linear, teamwork adds a residual delta to the base output. Weights are stacked
per teammate: `down[T, in, r]`, `up[T, r, out]` (no bias for Klein — base `disable_bias`).
**No alpha/rank scale factor** — delta is `up @ hidden` directly.

**With communication** (`lora_communication=True`, the released checkpoints):
```
hidden[pos] = Σ_teammates  down[teammate] · x[token at (teammate, pos)]     # shared low-rank bottleneck
delta[token at (teammate t, pos)] = up[t] · hidden[pos]
```
i.e. down-projections of all teammates at the same spatial position are summed into one
shared rank-r hidden, then each teammate reads it back through its own `up`. Summation is
order-independent (validated: numpy reproduces the golden `to_q` delta at mean-abs-err
0.0086 vs magnitude 0.817; no-communication variant is 63× worse).

### Sequence-axis realization (this is the whole design)
The adapter is given a per-token layout and computes, per token n:
- `teammate_id[n]` — which stacked slice `n` uses.
- `position_id[n]` — the shared logical position for communication.

Communication = **scatter-add by `position_id` then gather** (`hidden = Cᵀ C · lx`, never a
dense NxN matmul). For the regular Klein layout this collapses to `reshape [feat, T·L] →
[feat, L, T] → sum over T → broadcast` (the fast path). Text tokens get **unique**
position_ids so they only self-communicate.

### Teammate/position assignment for FLUX2-Klein
Sequence order = teammate-index order = **[edited(t0), source(t1), mask(t2)]** = sd.cpp's
target-then-refs order, so image block i uses `stack[i]` — **no remap** (unlike Python's
batch-axis component order [source,mask,edited]).
- **Double blocks:** `x` to `img_attn.qkv`/`img_mlp` is the **image stream only** (T contiguous
  blocks of L). Text is a separate, un-adapted stream (profile never touches `add_*_proj`) →
  no teammate.
- **Single blocks:** `x` to `linear1`/`linear2` is `[text(n_txt) | edited(L) | source(L) | mask(L)]`.
  Text tokens → **teammate index 1** (= `source`, the first `.in`/first-component). This
  reproduces a known upstream hack (single-block text is routed through the adapted fused
  projection and joint-attn dedups it to `first_component_per_batch`). `TeamworkConfig.text_teammate_index`
  holds this (currently 1). If Sam later fixes the hack, switch text to a synthetic zero teammate.

### Parity caveat (IMPORTANT for validation)
**Validate on IMAGE tokens only.** Image-token deltas should match the Python golden dump
within epsilon (communication sum is order-independent). **Text-token deltas will NOT match**:
Python keeps T divergent text copies and sums them then applies up[1]; sd.cpp keeps one text
copy with down[1]/up[1] and no cross-teammate mixing. This divergence is expected/accepted.

### Fused-layer handling
- Double-block `img_attn.qkv` is fused (out = [q|k|v] on feature axis) but the checkpoint has
  **separate** `to_q/to_k/to_v` stacks. The adapter must produce `delta = concat_feature(q_delta,
  k_delta, v_delta)`, each computed with its own down/up + communication. (`parse_teamwork_key`
  returns `slice`/`num_slices` for this.)
- Single-block `linear1` (=`to_qkv_mlp_proj`) and `linear2` (=`to_out`) are **already fused** in
  the checkpoint → 1:1, no sub-slicing. Same for `img_mlp.0/2` (=`ff.linear_in/out`) and modulation.

### Modulation (deferred to M3 — the one place that needs flux.hpp edits)
`double_stream_modulation_img.lin` / `single_stream_modulation.lin` take the shared timestep
vector (no token axis) but Teamwork wants **per-teammate** modulation. Output becomes
`[mod_dim, batch, T]` and `modulate()` must apply each teammate's shift/scale/gate to its own
token block. Klein `share_modulation=True` → one injection point. **Fallback:** skip
modulation adaptation in the first pass (4 tensors) and measure quality, then add it.

---

## 5. Naming: base vs teamwork (why we need a new map)

- BFL top-level `flux-2-klein-4b.safetensors` + GGUFs use **original/BFL naming** =
  sd.cpp internal (`double_blocks.N.img_attn.qkv`, `single_blocks.N.linear1`,
  `double_stream_modulation_img.lin`). That's why the base runs with no diffusers map.
- **Diffusers naming** lives only in the `transformer/` subdir. sd.cpp's diffusers block-map
  (`name_conversion.cpp` ~534 `convert_diffusers_dit_to_original_flux`) is **FLUX1-only** →
  sd.cpp currently has *no* diffusers-format Klein support.
- The **teamwork checkpoint is diffusers-named** → needs the FLUX2 map. Implemented as
  `parse_teamwork_key()` in `src/model/adapter/teamwork.hpp` (self-contained, not routed
  through name_conversion, to isolate teamwork's stacked/sub-slice semantics).

### Mapping table (teamwork diffusers key → sd.cpp internal, ALL 169 tensors validated)
| teamwork key (strip `.adapter.{down,up}`) | internal | slice |
|---|---|---|
| `transformer_blocks.N.attn.to_q/k/v` | `double_blocks.N.img_attn.qkv` | 0 / 1 / 2 of 3 |
| `transformer_blocks.N.attn.to_out.0` | `double_blocks.N.img_attn.proj` | — |
| `transformer_blocks.N.ff.linear_in` | `double_blocks.N.img_mlp.0` | — |
| `transformer_blocks.N.ff.linear_out` | `double_blocks.N.img_mlp.2` | — |
| `single_transformer_blocks.N.attn.to_qkv_mlp_proj` | `single_blocks.N.linear1` | — |
| `single_transformer_blocks.N.attn.to_out` | `single_blocks.N.linear2` | — |
| `double_stream_modulation_img.linear` | `double_stream_modulation_img.lin` | — |
| `single_stream_modulation.linear` | `single_stream_modulation.lin` | — |
| `*.adapter` (scalar) | joint-attn marker, ignored | — |
Counts: 5 double blocks × 6 + 20 single blocks × 2 + 2 modulation = 72 down + 72 up + 25 markers = 169.

Internal FLUX2 model detail (from GGUF, 149 tensors): `hidden=3072, heads=24, depth=5
double, 20 single, in_channels=128 (patch_size 1; = diffusers 32ch × 2×2 patch),
axes_dim={32,32,32,32}, theta=2000, ref_index_scale=10, share_modulation, disable_bias,
mlp SiLU`. Single-block `linear1` out = 27648 = qkv(9216) + gated-mlp(18432); `linear2`
in = 12288 = attn(3072)+mlp(9216).

---

## 6. sd.cpp integration points

- **WeightAdapter hook** (`core/ggml_extend.hpp` ~1649): every `Linear`/`Conv` forward routes
  through `ctx->weight_adapter->forward_with_lora(ctx, x, w, b, prefix, forward_params)` keyed
  by the module's dotted `prefix`. `MultiLoraAdapter` (`model/adapter/lora.hpp` ~897) is the
  existing impl; `TeamworkAdapter` is a sibling. Attach via `runner->set_weight_adapter(...)`
  (~2853); it's copied into every `GGMLRunnerContext` (~2693).
- `forward_with_lora` runs **per linear** and has NO layout info → `TeamworkAdapter` holds a
  `TeamworkLayout {n_txt, T, text_teammate_index, communication, rank}` set before each graph
  build; it infers per-token teammate/position ids from `x->ne[1]`, the prefix
  (`double_blocks`/`single_blocks`), and n_txt.
- **Debugging:** `GGMLRunnerContext::capture_tensor(name, t)` + `debug_tensors` can dump
  intermediate tensors — use this to compare per-layer deltas against the golden dump.
- **Ref-image flow** (reuse unchanged): `stable-diffusion.cpp` ~4118-4174 encodes ref images →
  `ref_latents`; `flux.hpp` `build_graph`/`forward_flux_chroma` concat on `ne[1]` + `gen_flux_pe`
  RoPE. Extra teammate tokens = extra ref latents; extra tokens sliced off output at ~1327.

---

## 7. Parity harness (golden reference)

- Script: `<scratchpad>/teamwork_parity_dump.py` (standalone; monkeypatches teamwork modules;
  loads local base + checkpoint offline via `Flux2KleinPipeline.from_pretrained` +
  `TeamworkPipeline.from_checkpoint(..., base_pipeline=...)`).
- Dump: `<scratchpad>/teamwork_parity/{tensors.safetensors, manifest.json, edited.png}`.
  256px apple→blue, seed 42, 4 steps. Block-0 layer I/O (`x/base/delta/final`) for double
  `to_q/k/v`, single `to_qkv_mlp_proj`, modulation, attn outputs, + e2e latents.
- `<scratchpad>` = `/tmp/claude-1000/-mnt-localssd-stable-diffusion-cpp/<session>/scratchpad`
  (**note: /tmp scratchpad may not persist — regenerate with the script if missing**).
- Layout facts: Python **batch-axis** order = `[source, mask, edited]` = teammates `[1,2,0]`
  (`selection.teammate_indices`). Joint-attn dense-seq order = teammate-index
  `[text, edited(t0,rope0), source(t1,rope10), mask(t2,rope20)]` = sd.cpp order. n_text=512,
  L_img=256 (@256px), T=3. `base+delta==final` exactly. `attn_keep=None` (all-to-all).

---

## 8. Progress (jj revisions, colocated jj+git repo)

Done & validated against the real checkpoint:
- `teamwork(1/n)`: `read_safetensors_metadata()` in `model_io/safetensors_io.{h,cpp}` (was skipped). ✅ builds.
- `teamwork(2/n)`: `TeamworkConfig` in `model/adapter/teamwork.hpp` — parses metadata. ✅
  (`FLUX2_PLUSATTN, T=3, rank=64, comm=1, text_teammate=1`, roster+ids correct).
- `teamwork(3/n)`: `parse_teamwork_key()` in `teamwork.hpp` — diffusers→internal FLUX2 map.
  ✅ all 169 tensors map, 0 unrecognized.
- `teamwork(5/n)`: `TeamworkModel` (GGMLRunner) loader + `TeamworkAdapter` (WeightAdapter) in
  `model/adapter/teamwork_model.hpp`; `--teamwork <ckpt>` CLI flag (`sd_ctx_params_t.teamwork_path`,
  common.cpp/.h, defaults/logging); loaded + attached in `stable-diffusion.cpp` (`load_teamwork()`,
  `attach_teamwork_adapter()` re-asserted at end of `apply_loras` since it shares the diffusion
  model's single weight_adapter slot with LoRAs — mutually exclusive with LoRA for now). ✅ END-TO-END:
  `--teamwork` loads the checkpoint (62 layers / 72 projections), attaches, graph builds + runs,
  produces a correct image. Delta is currently a **no-op** (`TeamworkLayout.valid=false`) — the
  adapter routes every linear through forward_with_lora but returns nullptr delta, so output ==
  base. M3 wires the real layout to activate it. GOTCHA fixed: don't do a 2nd load_tensors pass;
  register_param_tensors+prepare_params load the data (like LoraModel). Modulation layers are
  loaded but skipped in get_out_diff (M3).
- `teamwork(4/n)`: `sd_teamwork_lora_delta()` GGML kernel in `model/adapter/teamwork_adapter.hpp`
  + CPU unit test `tests/teamwork_kernel_test.cpp`. ✅ PASS (max err 1.975e-06 vs C++ reference,
  with N=2 batch + text tokens + communication). This is the validated M2 core (per-teammate
  down → shared-hidden sum → per-teammate up, teammate-major image blocks, text→teammate slice).
  Build the test with the §3 standalone recipe (ggml libs only; add `-Iggml/include` for `ggml-cpu.h`).

- `teamwork(5/n, folded)`: **M3 layout wiring — delta now ACTIVE & validated.** Added
  `WeightAdapter::set_sequence_layout(n_ref_latents, n_txt)` virtual hook (`ggml_extend.hpp`, no-op
  default; LoRA ignores it). `FluxRunner::build_graph` (flux.hpp ~1511) calls it each build with
  `ref_latents.size()` + `context->ne[1]`; `TeamworkAdapter::set_sequence_layout` builds a valid
  `TeamworkLayout` from config (T, text_teammate, communication) and sanity-checks `n_ref+1==T`.
  **GOTCHA FIXED:** the runner keys linears by FULL path `model.diffusion_model.double_blocks.0.
  img_attn.qkv.` but our layer map uses bare internal names → `get_out_diff` now strips the
  `model.diffusion_model.` prefix (was silently returning nullptr → bit-identical to base). ✅
  VALIDATED: (a) token layout arrives exactly as kernel assumes — DOUBLE x=[3072,768] (image-only,
  n_txt=0, 768=T·L=3·256), SINGLE x=[3072,1280] (512 text + 768 image); (b) delta active — teamwork
  output vs base-no-teamwork: 100% pixels changed, mean |Δ|=9.6/255; (c) blue-apple edit matches
  golden `edited.png` closely. Pixel-level e2e vs golden (~19 MAE) is a WEAK metric (Q8 quant +
  different VAE/sampler path), barely better than base — do NOT use it as the parity bar; use
  per-layer delta capture (M4).
  **PER-LAYER PARITY (double-block attn):** offline check `<scratchpad>/teamwork_delta_parity.py`
  reproduces the golden block-0 delta from checkpoint down/up + golden x using the sd.cpp teammate
  order [edited=t0, source=t1, mask=t2] with golden batch↔teammate remap [1,2,0]: to_q rel=1.05%,
  to_k=0.81%, to_v=2.34% (residual = bf16 rounding). WRONG-order control = 123% → the remap is
  required & correct. Transitively closes correctness: kernel==numpy(1.9e-6, M2) → numpy==golden(~1%)
  → sd.cpp feeds kernel exact layout (768/1280) in that order. Single-block `to_qkv_mlp_proj` +
  modulation deltas still un-checked (M4 leftovers).

- `teamwork(6/n)`: **per-teammate modulation — DONE & validated (M3 complete).** The shared
  modulation linears (`double_stream_modulation_img.lin`, `single_stream_modulation.lin`) take the
  timestep vec (no token axis) but Teamwork wants per-teammate shift/scale/gate. Implementation
  REUSES the validated kernel: `get_out_diff` replicates the vec to T "tokens" (L=1, teammate-major,
  n_txt=0) → `sd_teamwork_lora_delta` yields per-teammate delta `[out,T]`; `forward_with_lora`
  broadcasts the shared base `[out,1]` to `[out,T]` before adding. `Modulation::forward` reshapes on
  `out->ne[1]` (=T) so ModulationOut shift/scale/gate become `[dim,T]`. New guarded helpers in
  flux.hpp — `build_teammate_param` / `modulate_tw` / `gate_tw` — expand `[dim,T]` to per-token
  ([text→text_teammate | T image blocks]) and apply; they fall back to standard shared `modulate()`
  when teamwork inactive or param not per-teammate (`ne[1]!=T`, e.g. un-adapted double-block txt).
  New `WeightAdapter::get_teammate_modulation(T,n_txt,text_teammate)` hook (false default) gates it.
  Applied at 4 double-block img sites (n_txt=0) + 2 single-block sites (n_txt=layout.n_txt). Double
  txt stream stays shared (never adapted). **PARITY (offline, `teamwork_delta_parity.py`):** double
  modulation all 6 chunks rel 0.7–6.2% ✓; single shift/scale 0.8%/1.6% ✓. Single GATE chunk shows
  33% BUT that is a golden-dump artifact, NOT a bug: golden delta = `bf16(base+lora)-bf16(base)` and
  the FLUX2 gate base ≈10.8 so its bf16 ULP (0.059) is 1.33× the delta (0.044) → the reference delta
  there is rounding noise (ulp/|delta|>0.5 flagged in the script). The C++ path is provably identical
  to Python `teamwork/linear.py` (comm sum → per-teammate up with `sel()`=teammate_indices remap).
  e2e still a clean blue-apple edit. Non-teamwork models unaffected (all paths guarded).

Milestones: M0 substrate ✅ | M1 loading ✅ | M2 engine ✅ (kernel validated) | **M3 ✅ COMPLETE**
(layout wiring + per-teammate modulation, delta active & per-layer parity validated for double-attn
1–2%, modulation double 0.7–6% / single shift-scale <2%; single-gate golden is bf16-noise so
skipped) | M4 e2e parity vs golden (single-block `to_qkv_mlp_proj` capture still un-done; pixel e2e
too noisy) | M5 generality.

### M3 layout wiring — DONE ✅ (see teamwork(6/n) above)
Layout is wired via `WeightAdapter::set_sequence_layout` (called from `FluxRunner::build_graph`) →
`TeamworkAdapter::set_sequence_layout` → `TeamworkModel::set_layout({valid=true,...})`. Delta is
active; token layout validated (DOUBLE 768 image-only, SINGLE 512+768). Prefix-strip gotcha fixed.

### START HERE (next session): M4 parity + generality
M3 is COMPLETE (layout wiring + per-teammate modulation, both parity-validated — see teamwork(5/6 n)
above). Remaining:
1. **M4 single-block delta parity** (the one un-checked adapted layer type; the real correctness bar,
   pixel e2e is too noisy — see teamwork(6/n)).
   Use `GGMLRunnerContext::capture_tensor` to dump C++ block-0 deltas for `double_blocks.0.img_attn.qkv`
   (split into q/k/v, each [3072,768]) and `single_blocks.0.linear1`, then compare to golden
   `double.block0.attn.to_{q,k,v}.delta` [3,256,3072] and `single.block0...to_qkv_mlp_proj.delta`.
   REMAP: C++ seq order [edited(t0),source(t1),mask(t2)] ↔ golden batch order [source,mask,edited]
   (rows [1,2,0]). Validate IMAGE tokens only; text-token deltas expected to differ (§4 caveat).
   CAVEAT: base is Q8_0 quantized in the C++ run so x differs from bf16 golden → expect ~1% not
   bit-exact; for a clean kernel check, feed golden `to_q.x` through the kernel offline instead.
- CAVEAT: teamwork + LoRA share the diffusion weight_adapter slot (mutually exclusive now).

---

## 9. Remaining plan

### M1 finish — `TeamworkAdapter` load side (next)
- New `TeamworkAdapter : WeightAdapter` (in `teamwork.hpp` or `teamwork.cpp`). Load the stacked
  `[T,·,·]` down/up tensors from the checkpoint into GGML (mirror `LoraModel::load_from_file`
  in `lora.hpp` ~44; uses `model_manager->loader()`), keyed by internal layer via
  `parse_teamwork_key`. Group sub-slices (q/k/v) under one layer entry. Hold `TeamworkConfig`.
- Add `--teamwork <ckpt>` CLI flag (`examples/common/common.cpp` + `include/stable-diffusion.h`
  `sd_ctx_params_t`), a loader in `stable-diffusion.cpp` (near lora wiring ~1496-1546) that
  detects `modelspec.type==teamwork`, builds the adapter, `set_weight_adapter`. Log config.

### M2 — communicating-LoRA forward (the core)
- Implement `forward_with_lora`: `base(x) + delta`. Per adapted layer, using the layout:
  loop T teammates (small): slice x block → `x_i @ down[i]` → scatter-add by position → shared
  hidden → `hidden @ up[i]` → write back. Handle qkv sub-slice concat; single-block text→teammate1;
  double vs single stream detection by prefix.
- Validate with `capture_tensor` vs golden dump **image-token** deltas (double `to_q`, single
  `to_qkv_mlp_proj`) within epsilon. Text deltas expected to differ.

### M3 — wiring + modulation
- Thread `TeamworkLayout` (T, L, n_txt, per-token ids, present/attn_keep) from the ref-latent
  structure into the adapter each graph build. Assert token order == teammate/id order.
- Per-teammate modulation: touch `flux.hpp` modulate() path (see §4). Klein = one injection point.

### M4 — end-to-end example + parity
- Small path/CLI: base Klein + `--teamwork` + source.png + mask.png + prompt → edited.png
  (harness maps teammate-name → ref order: refs = [source, mask]). Compare to golden `edited.png`
  + latents within epsilon. Bar: visually correct + kernel-tolerance, not bit-exact.

### Deferred (YAGNI, noted in `teamwork_adapter.hpp`)
- **General per-token teammate/position ids** (arbitrary token→teammate + scatter/gather
  communication): not built — no motivating case, scatter/gather slow. The kernel hard-codes
  the `[text | T equal image blocks]` ViT layout. Revisit for irregular/interleaved teammates.
- **Matmul fusion:** the T looped down/up matmuls can become batched matmuls over the teammate
  axis (`[in,L,T,N]` + ggml ne2 batching); communicating down+sum can collapse to one matmul via
  concatenated teammate features (`D_cat [in*T, r]`). Benefit scales with T (modest at T=3, real
  for large-T SVBRDF); adds reshape/permute cost — benchmark first.

### M5 — generality (later)
- `attn_allow` masking; teammate present/deactivation; FLUX1_PLUSATTN; text2img RGB→X/SVBRDF
  (true new-image teammates + synthetic zero text teammate); `BatchBuilder`-style helper; full
  `--teammate-input name=f`/`--teammate-output name=f` CLI; `overpainteditor` `InferenceBackend`.

---

## 10. Resolved design decisions (don't relitigate)
- Sequence axis (not batch/channel); ViTs only.
- Target = FLUX2_PLUSATTN/Klein-4B first; joint attention is inherent to the sequence layout
  (all-to-all here since `attn_allow=None`, all present).
- Text: double-block = un-adapted separate stream; single-block = teammate index 1 (source).
- Teammate-index sequence order = no remap. Communication = position-id scatter/gather (reshape
  fast path for regular layout). No alpha/rank scale.
- `--teamwork <ckpt>` flag (not overloading `--lora`); `TeamworkAdapter` beside `MultiLoraAdapter`.
- Validate on image tokens only.
