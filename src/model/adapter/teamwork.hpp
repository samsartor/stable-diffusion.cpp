#ifndef __SD_MODEL_ADAPTER_TEAMWORK_HPP__
#define __SD_MODEL_ADAPTER_TEAMWORK_HPP__

#include <map>
#include <string>
#include <vector>

#include "core/util.h"
#include "model_io/safetensors_io.h"

// Configuration for a "teamwork" adapter (https://github.com/samsartor/teamwork).
// Everything travels in the checkpoint's safetensors "__metadata__" map; the
// trainable tensors are just the per-teammate LoRA stacks.
struct TeamworkConfig {
    bool valid = false;

    std::string base_model;              // e.g. "black-forest-labs/FLUX.2-klein-4B"
    std::string profile;                 // e.g. "FLUX2_PLUSATTN"

    std::vector<std::string> teammates;  // e.g. ["edited.out", "source.in", "mask.in"]
    std::vector<bool> teammate_is_input; // true for ".in" (pinned) teammates
    std::vector<int> teammate_image_ids; // per-teammate RoPE T-axis offset (e.g. 0,10,20)

    int lora_rank           = 0;
    bool lora_communication = true;   // shared low-rank bottleneck across teammates
    bool use_bias           = false;
    std::string attn_allow;           // optional row-major "100;011;..." cross-teammate attn mask

    // Teammate whose LoRA adapts single-block text tokens. In the current
    // FLUX2_PLUSATTN checkpoints text is routed through the first input teammate
    // (a known upstream hack); we reproduce that faithfully. -1 => none.
    int text_teammate_index = -1;

    int num_teammates() const {
        return (int)teammates.size();
    }

    static bool parse_bool(const std::string& s, bool fallback = false) {
        if (s.empty()) {
            return fallback;
        }
        std::string l = s;
        for (auto& c : l) {
            c = (char)std::tolower((unsigned char)c);
        }
        return l == "true" || l == "1" || l == "yes";
    }

    static TeamworkConfig from_metadata(const std::map<std::string, std::string>& md) {
        TeamworkConfig cfg;
        auto get = [&](const std::string& k) -> std::string {
            auto it = md.find(k);
            return it == md.end() ? std::string() : it->second;
        };

        // A teamwork checkpoint self-identifies via modelspec.type.
        if (get("modelspec.type") != "teamwork") {
            return cfg;  // valid = false
        }

        cfg.base_model         = get("base_model");
        cfg.profile            = get("teamwork.profile");
        cfg.lora_communication = parse_bool(get("teamwork.lora_communication"), true);
        cfg.use_bias           = parse_bool(get("teamwork.use_bias"), false);
        cfg.attn_allow         = get("teamwork.attn_allow");
        {
            std::string rank = get("teamwork.lora_rank");
            cfg.lora_rank    = rank.empty() ? 0 : std::stoi(rank);
        }

        for (const auto& t : split_string(get("teamwork.teammates"), ',')) {
            std::string name = trim(t);
            if (name.empty()) {
                continue;
            }
            bool is_input = ends_with(name, ".in");
            if (is_input && cfg.text_teammate_index < 0) {
                cfg.text_teammate_index = (int)cfg.teammates.size();
            }
            cfg.teammates.push_back(name);
            cfg.teammate_is_input.push_back(is_input);
        }

        for (const auto& id : split_string(get("teamwork.teammate_image_ids"), ',')) {
            std::string v = trim(id);
            if (!v.empty()) {
                cfg.teammate_image_ids.push_back(std::stoi(v));
            }
        }
        // Default: all teammates aligned at offset 0 if unspecified.
        if (cfg.teammate_image_ids.empty()) {
            cfg.teammate_image_ids.assign(cfg.teammates.size(), 0);
        }

        cfg.valid = !cfg.teammates.empty() && cfg.lora_rank > 0 &&
                    cfg.teammate_image_ids.size() == cfg.teammates.size();
        return cfg;
    }

    static TeamworkConfig from_checkpoint(const std::string& path) {
        std::map<std::string, std::string> md;
        std::string error;
        if (!read_safetensors_metadata(path, md, &error)) {
            LOG_WARN("teamwork: failed to read metadata from '%s': %s", path.c_str(), error.c_str());
            return TeamworkConfig();
        }
        return from_metadata(md);
    }

    void log() const {
        if (!valid) {
            LOG_WARN("teamwork: invalid/unrecognized config");
            return;
        }
        std::string roster;
        for (size_t i = 0; i < teammates.size(); ++i) {
            roster += (i ? ", " : "") + teammates[i] + "(id=" + std::to_string(teammate_image_ids[i]) + ")";
        }
        LOG_INFO("teamwork: profile=%s base=%s T=%d rank=%d comm=%d bias=%d text_teammate=%d",
                 profile.c_str(), base_model.c_str(), num_teammates(), lora_rank,
                 (int)lora_communication, (int)use_bias, text_teammate_index);
        LOG_INFO("teamwork: teammates = [%s]", roster.c_str());
    }
};

// Parsed identity of a teamwork checkpoint tensor key.
//
// Teamwork keys use the diffusers module tree, e.g.
//   transformer_blocks.0.attn.to_q.adapter.down
//   single_transformer_blocks.0.attn.to_qkv_mlp_proj.adapter.up
//   double_stream_modulation_img.linear.adapter.down
//   single_transformer_blocks.0.adapter                 (scalar joint-attn marker)
// We map the layer path onto sd.cpp's internal FLUX2 names. Only the double-block
// fused qkv needs sub-slicing (diffusers keeps to_q/to_k/to_v separate; sd.cpp
// fuses them into img_attn.qkv), everything else is 1:1.
struct TeamworkKeyInfo {
    bool ok       = false;  // recognized adapter tensor
    bool marker   = false;  // ".adapter" scalar marker (ignored)
    std::string layer;      // internal module path, e.g. "double_blocks.0.img_attn.qkv"
    int slice     = -1;     // sub-slice index within a fused output (-1 = whole tensor)
    int num_slices = 1;     // number of sub-slices the fused output splits into
    enum Param { DOWN,
                 UP,
                 BIAS } param = DOWN;
};

// Match "<prefix><int>." at the start of `name`; on success fill `idx` and `rest`.
static inline bool tw_match_block(const std::string& name, const std::string& prefix, int& idx, std::string& rest) {
    if (!starts_with(name, prefix)) {
        return false;
    }
    size_t p = prefix.size();
    size_t q = name.find('.', p);
    if (q == std::string::npos || q == p) {
        return false;
    }
    for (size_t i = p; i < q; ++i) {
        if (!isdigit((unsigned char)name[i])) {
            return false;
        }
    }
    idx  = std::stoi(name.substr(p, q - p));
    rest = name.substr(q + 1);
    return true;
}

static inline TeamworkKeyInfo parse_teamwork_key(const std::string& key) {
    TeamworkKeyInfo info;

    // Strip the trailing ".adapter[.<param>]" suffix to recover the layer path.
    std::string layer;
    if (ends_with(key, ".adapter.down")) {
        info.param = TeamworkKeyInfo::DOWN;
        layer      = key.substr(0, key.size() - std::string(".adapter.down").size());
    } else if (ends_with(key, ".adapter.up")) {
        info.param = TeamworkKeyInfo::UP;
        layer      = key.substr(0, key.size() - std::string(".adapter.up").size());
    } else if (ends_with(key, ".adapter.bias")) {
        info.param = TeamworkKeyInfo::BIAS;
        layer      = key.substr(0, key.size() - std::string(".adapter.bias").size());
    } else if (ends_with(key, ".adapter")) {
        info.marker = true;
        info.ok     = true;  // recognized, but nothing to load
        return info;
    } else {
        return info;  // not a teamwork tensor
    }

    int n;
    std::string rest;
    if (tw_match_block(layer, "single_transformer_blocks.", n, rest)) {
        std::string base = "single_blocks." + std::to_string(n) + ".";
        if (rest == "attn.to_qkv_mlp_proj") {
            info.layer = base + "linear1";  // fused qkv+mlp-in (1:1, pre-fused in ckpt)
        } else if (rest == "attn.to_out") {
            info.layer = base + "linear2";  // fused attn-out + mlp-out
        } else {
            return info;
        }
    } else if (tw_match_block(layer, "transformer_blocks.", n, rest)) {
        std::string base = "double_blocks." + std::to_string(n) + ".";
        if (rest == "attn.to_q") {
            info.layer = base + "img_attn.qkv";
            info.slice = 0;
            info.num_slices = 3;
        } else if (rest == "attn.to_k") {
            info.layer = base + "img_attn.qkv";
            info.slice = 1;
            info.num_slices = 3;
        } else if (rest == "attn.to_v") {
            info.layer = base + "img_attn.qkv";
            info.slice = 2;
            info.num_slices = 3;
        } else if (rest == "attn.to_out.0") {
            info.layer = base + "img_attn.proj";
        } else if (rest == "ff.linear_in") {
            info.layer = base + "img_mlp.0";
        } else if (rest == "ff.linear_out") {
            info.layer = base + "img_mlp.2";
        } else {
            return info;
        }
    } else if (layer == "double_stream_modulation_img.linear") {
        info.layer = "double_stream_modulation_img.lin";
    } else if (layer == "double_stream_modulation_txt.linear") {
        info.layer = "double_stream_modulation_txt.lin";
    } else if (layer == "single_stream_modulation.linear") {
        info.layer = "single_stream_modulation.lin";
    } else {
        return info;
    }

    info.ok = true;
    return info;
}

#endif  // __SD_MODEL_ADAPTER_TEAMWORK_HPP__
