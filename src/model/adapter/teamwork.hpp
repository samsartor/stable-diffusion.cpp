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

#endif  // __SD_MODEL_ADAPTER_TEAMWORK_HPP__
