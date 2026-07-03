#ifndef __SD_MODEL_ADAPTER_TEAMWORK_MODEL_HPP__
#define __SD_MODEL_ADAPTER_TEAMWORK_MODEL_HPP__

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/ggml_extend.hpp"
#include "model/adapter/teamwork.hpp"
#include "model/adapter/teamwork_adapter.hpp"
#include "model_loader.h"
#include "model_manager.h"

// Per-forward sequence-axis layout, set before each diffusion graph build (M3 wires
// this from the ref-latent structure). See TEAMWORK_PLAN.md §4.
struct TeamworkLayout {
    int T             = 1;      // number of image teammates (= contiguous image blocks)
    int n_txt         = 0;      // text tokens prefixing single-block streams
    int text_teammate = 0;      // teammate slice for single-block text tokens
    bool communication = true;  // shared low-rank bottleneck across teammates
    bool valid         = false;
};

// Loads a teamwork checkpoint's per-teammate LoRA stacks and computes the per-layer
// communicating-LoRA delta at runtime. Mirrors LoraModel's GGMLRunner load path.
struct TeamworkModel : public GGMLRunner {
    struct SubProj {
        int slice      = -1;
        int num_slices = 1;
        ggml_tensor* down = nullptr;  // ne=[r, in, T]
        ggml_tensor* up   = nullptr;  // ne=[out, r, T]
    };
    struct Layer {
        std::vector<SubProj> subs;  // ordered by slice (fused qkv: q,k,v)
    };

    std::string file_path;
    std::shared_ptr<ModelManager> model_manager;
    ggml_backend_t params_backend = nullptr;
    bool load_failed              = false;
    TeamworkConfig config;
    TeamworkLayout layout;

    std::unordered_map<std::string, ggml_tensor*> raw_tensors;  // raw checkpoint key -> tensor
    std::map<std::string, Layer> layers;                        // internal module path -> grouped subs

    TeamworkModel(ggml_backend_t backend,
                  ggml_backend_t params_backend_,
                  const std::string& file_path,
                  std::shared_ptr<ModelManager> manager = std::make_shared<ModelManager>())
        : GGMLRunner(backend, manager), file_path(file_path), model_manager(std::move(manager)), params_backend(params_backend_) {
        config = TeamworkConfig::from_checkpoint(file_path);
        // Raw keys (no name conversion); we map them ourselves via parse_teamwork_key.
        if (model_manager == nullptr || !model_manager->loader().init_from_file(file_path, "")) {
            load_failed = true;
        }
    }

    std::string get_desc() override {
        return "teamwork";
    }

    bool load_from_file(int n_threads) {
        LOG_INFO("loading teamwork adapter from '%s'", file_path.c_str());
        if (load_failed) {
            LOG_ERROR("init teamwork loader from file failed: '%s'", file_path.c_str());
            return false;
        }
        if (!config.valid) {
            LOG_ERROR("'%s' is not a valid teamwork checkpoint", file_path.c_str());
            return false;
        }
        config.log();

        std::unordered_map<std::string, TensorStorage> tensors_to_create;
        std::mutex mtx;
        bool dry_run          = true;
        auto on_new_tensor_cb = [&](const TensorStorage& ts, ggml_tensor** dst) -> bool {
            if (dry_run) {
                TeamworkKeyInfo info = parse_teamwork_key(ts.name);
                if (!info.ok || info.marker || info.param == TeamworkKeyInfo::BIAS) {
                    return true;  // skip markers / unrecognized / (unused) bias
                }
                std::lock_guard<std::mutex> lock(mtx);
                tensors_to_create[ts.name] = ts;
            } else {
                auto it = raw_tensors.find(ts.name);
                if (it != raw_tensors.end()) {
                    *dst = it->second;
                }
            }
            return true;
        };

        model_manager->set_n_threads(n_threads);
        ModelLoader& loader = model_manager->loader();
        loader.load_tensors(on_new_tensor_cb);  // dry run: collect

        if (tensors_to_create.empty()) {
            LOG_ERROR("teamwork checkpoint '%s' had no adapter tensors", file_path.c_str());
            return false;
        }
        for (const auto& [name, ts] : tensors_to_create) {
            raw_tensors[name] = ggml_new_tensor(params_ctx, ts.type, ts.n_dims, ts.ne);
        }

        std::map<std::string, ggml_tensor*> tensors(raw_tensors.begin(), raw_tensors.end());
        if (!model_manager->register_param_tensors("Teamwork",
                                                   std::move(tensors),
                                                   ModelManager::ResidencyMode::ParamBackend,
                                                   runtime_backend,
                                                   params_backend) ||
            !model_manager->validate_registered_tensors()) {
            LOG_ERROR("teamwork model manager registration failed");
            return false;
        }
        std::vector<ggml_tensor*> params;
        params.reserve(raw_tensors.size());
        for (const auto& [name, t] : raw_tensors) {
            params.push_back(t);
        }
        // register_param_tensors + prepare_params load the tensor data from disk (as in
        // LoraModel); no second load_tensors pass is needed.
        if (!model_manager->prepare_params(params)) {
            LOG_ERROR("teamwork model manager prepare params failed");
            return false;
        }
        (void)dry_run;

        // Group the raw tensors into per-layer, per-sub-slice down/up pairs.
        std::map<std::string, std::map<int, SubProj>> tmp;
        for (const auto& [name, tensor] : raw_tensors) {
            TeamworkKeyInfo info = parse_teamwork_key(name);
            SubProj& sp   = tmp[info.layer][info.slice];
            sp.slice      = info.slice;
            sp.num_slices = info.num_slices;
            if (info.param == TeamworkKeyInfo::DOWN) {
                sp.down = tensor;
            } else if (info.param == TeamworkKeyInfo::UP) {
                sp.up = tensor;
            }
        }
        int n_layers = 0, n_subs = 0;
        for (auto& [layer, slices] : tmp) {
            Layer L;
            for (auto& [slice, sp] : slices) {  // std::map iterates slices ascending (q,k,v)
                if (sp.down == nullptr || sp.up == nullptr) {
                    LOG_WARN("teamwork layer '%s' slice %d missing down/up", layer.c_str(), slice);
                    continue;
                }
                L.subs.push_back(sp);
                n_subs++;
            }
            if (!L.subs.empty()) {
                layers[layer] = std::move(L);
                n_layers++;
            }
        }
        LOG_INFO("teamwork: loaded %d adapted layers (%d projections)", n_layers, n_subs);
        return true;
    }

    void set_layout(const TeamworkLayout& l) {
        layout = l;
    }

    // Communicating-LoRA delta for one adapted linear, keyed by its module prefix
    // (e.g. "double_blocks.0.img_attn.qkv."). Returns nullptr if not adapted.
    ggml_tensor* get_out_diff(ggml_context* ctx, ggml_tensor* x, const std::string& prefix) {
        if (!layout.valid) {
            return nullptr;
        }
        std::string layer = prefix;
        // Runner keys linears by full module path (e.g. "model.diffusion_model.double_blocks.0.
        // img_attn.qkv."); our layer map uses the bare internal names from parse_teamwork_key.
        const std::string kRunnerPrefix = "model.diffusion_model.";
        if (starts_with(layer, kRunnerPrefix)) {
            layer.erase(0, kRunnerPrefix.size());
        }
        if (!layer.empty() && layer.back() == '.') {
            layer.pop_back();
        }
        auto it = layers.find(layer);
        if (it == layers.end()) {
            return nullptr;
        }
        // Modulation linears take the shared timestep vector (x = SiLU(vec), [in, 1]) with
        // no token axis, but teamwork wants a per-teammate output. Replicate x to T "tokens"
        // (L=1, teammate-major, no text) so the same kernel yields the per-teammate delta
        // [out, T]; the flux modulate() path then applies each teammate's mod to its block.
        if (contains(layer, "modulation")) {
            const SubProj& sub = it->second.subs[0];  // modulation lin is a single fused proj
            ggml_tensor* xr    = ggml_repeat(ctx, x,
                                             ggml_new_tensor_3d(ctx, x->type, x->ne[0], layout.T, x->ne[2]));
            return sd_teamwork_lora_delta(ctx, xr, sub.down, sub.up,
                                          /*n_txt=*/0, layout.T, layout.text_teammate,
                                          layout.communication);
        }
        const bool is_single = starts_with(layer, "single_blocks.");
        const int n_txt      = is_single ? layout.n_txt : 0;

        ggml_tensor* delta = nullptr;
        for (const SubProj& sub : it->second.subs) {
            ggml_tensor* d = sd_teamwork_lora_delta(ctx, x, sub.down, sub.up,
                                                    n_txt, layout.T, layout.text_teammate,
                                                    layout.communication);
            delta = delta ? ggml_concat(ctx, delta, d, 0) : d;  // fused qkv: concat on feature axis
        }
        return delta;
    }
};

// WeightAdapter wrapping a TeamworkModel (sibling to MultiLoraAdapter). Adds the
// communicating-LoRA delta to each adapted linear's output; teamwork never merges
// into weights, so patch_weight is a no-op.
struct TeamworkAdapter : public WeightAdapter {
    std::shared_ptr<TeamworkModel> model;

    explicit TeamworkAdapter(std::shared_ptr<TeamworkModel> model) : model(std::move(model)) {}

    ggml_tensor* patch_weight(ggml_context*, ggml_backend_t, ggml_tensor* weight, const std::string&) override {
        return weight;
    }

    ggml_tensor* forward_with_lora(ggml_context* ctx,
                                   ggml_backend_t,
                                   ggml_tensor* x,
                                   ggml_tensor* w,
                                   ggml_tensor* b,
                                   const std::string& prefix,
                                   WeightAdapter::ForwardParams fp) override {
        ggml_tensor* out;
        if (fp.op_type == ForwardParams::op_type_t::OP_LINEAR) {
            out = ggml_ext_linear(ctx, x, w, b, fp.linear.force_prec_f32, fp.linear.scale);
        } else {
            out = ggml_ext_conv_2d(ctx, x, w, b,
                                   fp.conv2d.s0, fp.conv2d.s1, fp.conv2d.p0, fp.conv2d.p1,
                                   fp.conv2d.d0, fp.conv2d.d1, fp.conv2d.direct,
                                   fp.conv2d.circular_x, fp.conv2d.circular_y, fp.conv2d.scale);
        }
        ggml_tensor* diff = model->get_out_diff(ctx, x, prefix);
        if (diff != nullptr) {
            // Modulation delta is per-teammate [out, T] while the shared base out is [out, 1];
            // broadcast base over the teammate axis before adding. Token-axis deltas match 1:1.
            if (diff->ne[1] != out->ne[1]) {
                out = ggml_repeat(ctx, out, diff);
            }
            out = ggml_add(ctx, out, diff);
        }
        return out;
    }

    size_t get_extra_graph_size() override {
        return 10240 + model->raw_tensors.size() * 20;
    }

    bool get_teammate_modulation(int& T, int& n_txt, int& text_teammate) override {
        if (!model->layout.valid) {
            return false;
        }
        T             = model->layout.T;
        n_txt         = model->layout.n_txt;
        text_teammate = model->layout.text_teammate;
        return true;
    }

    // Activate the communicating-LoRA layout for this graph build. n_ref_latents comes
    // from the diffusion runner's ref-latent structure; teammate count is authoritative
    // from the checkpoint config (target + refs). n_txt is the text prefix single blocks
    // split on. See TEAMWORK_PLAN.md §M3.
    void set_sequence_layout(int n_ref_latents, int n_txt) override {
        const int T = model->config.num_teammates();
        if (n_ref_latents + 1 != T) {
            LOG_WARN("teamwork: ref_latents(%d)+1 != teammates(%d); layout may be wrong",
                     n_ref_latents, T);
        }
        TeamworkLayout l;
        l.T             = T;
        l.n_txt         = n_txt;
        l.text_teammate = model->config.text_teammate_index;
        l.communication = model->config.lora_communication;
        l.valid         = true;
        model->set_layout(l);
    }
};

#endif  // __SD_MODEL_ADAPTER_TEAMWORK_MODEL_HPP__
