#include "include/CDwarfStar.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *environment;
    const char *filename;
} afm_ds4_metal_source;

static int afm_ds4_configure_metal_sources(
        const char *root,
        char *error,
        size_t error_capacity) {
    static const afm_ds4_metal_source sources[] = {
        {"DS4_METAL_FLASH_ATTN_SOURCE", "flash_attn.metal"},
        {"DS4_METAL_DENSE_SOURCE", "dense.metal"},
        {"DS4_METAL_MOE_SOURCE", "moe.metal"},
        {"DS4_METAL_DSV4_HC_SOURCE", "dsv4_hc.metal"},
        {"DS4_METAL_UNARY_SOURCE", "unary.metal"},
        {"DS4_METAL_DSV4_KV_SOURCE", "dsv4_kv.metal"},
        {"DS4_METAL_DSV4_ROPE_SOURCE", "dsv4_rope.metal"},
        {"DS4_METAL_DSV4_MISC_SOURCE", "dsv4_misc.metal"},
        {"DS4_METAL_ARGSORT_SOURCE", "argsort.metal"},
        {"DS4_METAL_CPY_SOURCE", "cpy.metal"},
        {"DS4_METAL_CONCAT_SOURCE", "concat.metal"},
        {"DS4_METAL_GET_ROWS_SOURCE", "get_rows.metal"},
        {"DS4_METAL_SUM_ROWS_SOURCE", "sum_rows.metal"},
        {"DS4_METAL_SOFTMAX_SOURCE", "softmax.metal"},
        {"DS4_METAL_REPEAT_SOURCE", "repeat.metal"},
        {"DS4_METAL_GLU_SOURCE", "glu.metal"},
        {"DS4_METAL_NORM_SOURCE", "norm.metal"},
        {"DS4_METAL_BIN_SOURCE", "bin.metal"},
        {"DS4_METAL_SET_ROWS_SOURCE", "set_rows.metal"},
    };

    if (!root || !root[0]) {
        snprintf(error, error_capacity, "DwarfStar Metal source directory is missing");
        return -1;
    }

    for (size_t index = 0; index < sizeof(sources) / sizeof(sources[0]); index++) {
        char path[PATH_MAX];
        int length = snprintf(path, sizeof(path), "%s/%s", root, sources[index].filename);
        if (length < 0 || (size_t)length >= sizeof(path)) {
            snprintf(error, error_capacity, "DwarfStar Metal source path is too long");
            return -1;
        }
        if (setenv(sources[index].environment, path, 1) != 0) {
            snprintf(error, error_capacity, "Unable to configure %s: %s",
                     sources[index].environment, strerror(errno));
            return -1;
        }
    }
    return 0;
}

int afm_ds4_engine_open(
        ds4_engine **out,
        const char *model_path,
        int context_size,
        uint32_t prefill_chunk,
        int power_percent,
        const char *dspark_support_path,
        int dspark_draft_tokens,
        float dspark_confidence_threshold,
        int dspark_strict,
        const char *metal_source_root,
        char *error,
        size_t error_capacity) {
    if (!out || !model_path || !model_path[0]) {
        snprintf(error, error_capacity, "DwarfStar model path is missing");
        return -1;
    }
    if (afm_ds4_configure_metal_sources(metal_source_root, error, error_capacity) != 0) {
        return -1;
    }

    ds4_engine_options options = {0};
    options.model_path = model_path;
    options.backend = DS4_BACKEND_METAL;
    options.context_size = context_size;
    options.prefill_chunk = prefill_chunk;
    options.power_percent = power_percent;
    options.mtp_path = dspark_support_path;
    options.mtp_draft_tokens = dspark_draft_tokens > 0 ? dspark_draft_tokens : 5;
    options.mtp_margin = 3.0f;
    options.dspark = dspark_support_path && dspark_support_path[0];
    options.dspark_strict = dspark_strict != 0;
    options.dspark_confidence_threshold = dspark_confidence_threshold;
    options.dspark_confidence_threshold_set = options.dspark;

    if (ds4_engine_open(out, &options) != 0) {
        snprintf(error, error_capacity, "DwarfStar failed to open model %s", model_path);
        return -1;
    }
    return 0;
}

int afm_ds4_engine_open_mapped(
        ds4_engine **out,
        const char *metadata_path,
        uint64_t virtual_size,
        const ds4_model_map_region *regions,
        size_t region_count,
        int context_size,
        uint32_t prefill_chunk,
        int power_percent,
        const char *dspark_support_path,
        int dspark_draft_tokens,
        float dspark_confidence_threshold,
        int dspark_strict,
        const char *metal_source_root,
        char *error,
        size_t error_capacity) {
    if (!out || !metadata_path || !metadata_path[0] || !regions || region_count == 0) {
        snprintf(error, error_capacity, "DwarfStar AFM projection is incomplete");
        return -1;
    }
    if (afm_ds4_configure_metal_sources(metal_source_root, error, error_capacity) != 0) {
        return -1;
    }

    ds4_model_mapping mapping = {
        .virtual_size = virtual_size,
        .regions = regions,
        .region_count = region_count,
    };
    ds4_engine_options options = {0};
    options.model_path = metadata_path;
    options.model_mapping = &mapping;
    options.backend = DS4_BACKEND_METAL;
    options.context_size = context_size;
    options.prefill_chunk = prefill_chunk;
    options.power_percent = power_percent;
    options.mtp_path = dspark_support_path;
    options.mtp_draft_tokens = dspark_draft_tokens > 0 ? dspark_draft_tokens : 5;
    options.mtp_margin = 3.0f;
    options.dspark = dspark_support_path && dspark_support_path[0];
    options.dspark_strict = dspark_strict != 0;
    options.dspark_confidence_threshold = dspark_confidence_threshold;
    options.dspark_confidence_threshold_set = options.dspark;

    if (ds4_engine_open(out, &options) != 0) {
        snprintf(error, error_capacity, "DwarfStar failed to open AFM projection %s", metadata_path);
        return -1;
    }
    return 0;
}

void afm_ds4_tokens_init(ds4_tokens *tokens) {
    if (tokens) memset(tokens, 0, sizeof(*tokens));
}

void afm_ds4_free(void *pointer) {
    free(pointer);
}
