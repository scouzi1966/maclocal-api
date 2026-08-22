#include "include/CDwarfStar.h"
#include "../../vendor/ds4/ds4_kvstore.h"

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

void afm_ds4_tokens_init(ds4_tokens *tokens) {
    if (tokens) memset(tokens, 0, sizeof(*tokens));
}

void afm_ds4_free(void *pointer) {
    free(pointer);
}

struct afm_ds4_prefix_cache {
    ds4_kvstore store;
};

static void afm_ds4_prefix_cache_log(
        void *unused,
        ds4_kvstore_log_type type,
        const char *message) {
    (void)unused;
    const char *level = type == DS4_KVSTORE_LOG_WARNING ? "warning" : "info";
    fprintf(stderr, "[DwarfStarPrefixCache] %s: %s\n", level, message);
}

int afm_ds4_prefix_cache_open(
        afm_ds4_prefix_cache **out,
        const char *directory,
        uint64_t budget_mb,
        char *error,
        size_t error_capacity) {
    if (!out || !directory || !directory[0]) {
        snprintf(error, error_capacity, "DwarfStar prefix-cache directory is missing");
        return -1;
    }
    afm_ds4_prefix_cache *cache = calloc(1, sizeof(*cache));
    if (!cache) {
        snprintf(error, error_capacity, "Unable to allocate DwarfStar prefix cache");
        return -1;
    }
    ds4_kvstore_options options = ds4_kvstore_default_options();
    if (!ds4_kvstore_open(
            &cache->store,
            directory,
            budget_mb,
            true,
            options,
            "AFM",
            afm_ds4_prefix_cache_log,
            NULL)) {
        free(cache);
        snprintf(error, error_capacity, "Unable to open DwarfStar prefix cache at %s", directory);
        return -1;
    }
    *out = cache;
    return 0;
}

void afm_ds4_prefix_cache_close(afm_ds4_prefix_cache *cache) {
    if (!cache) return;
    ds4_kvstore_close(&cache->store);
    free(cache);
}

int afm_ds4_prefix_cache_restore(
        afm_ds4_prefix_cache *cache,
        ds4_engine *engine,
        ds4_session *session,
        ds4_tokens *prompt,
        char *error,
        size_t error_capacity) {
    if (!cache || !engine || !session || !prompt) return 0;
    size_t text_length = 0;
    char *text = ds4_kvstore_render_tokens_text(engine, prompt, &text_length);
    if (!text) return 0;

    ds4_tokens effective = {0};
    ds4_kvstore_load_result result = {0};
    int loaded = ds4_kvstore_try_load_text(
        &cache->store,
        engine,
        session,
        text,
        &effective,
        &result,
        NULL,
        false);
    free(text);
    ds4_kvstore_load_result_free(&result);

    if (loaded > 0) {
        ds4_tokens_free(prompt);
        *prompt = effective;
        return loaded;
    }
    ds4_tokens_free(&effective);
    if (loaded < 0) {
        snprintf(error, error_capacity, "DwarfStar prefix-cache restore failed");
        return -1;
    }
    return 0;
}

int afm_ds4_prefix_cache_store_session(
        afm_ds4_prefix_cache *cache,
        ds4_engine *engine,
        ds4_session *session,
        const char *reason,
        char *error,
        size_t error_capacity) {
    if (!cache || !engine || !session) return 0;
    const ds4_tokens *tokens = ds4_session_tokens(session);
    if (!tokens || tokens->len <= 0) return 0;
    return ds4_kvstore_store_live_prefix(
        &cache->store,
        engine,
        session,
        tokens,
        tokens->len,
        reason ? reason : "continued",
        NULL,
        error,
        error_capacity) ? tokens->len : 0;
}
