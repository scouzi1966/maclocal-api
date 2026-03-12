#pragma once
#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void* tokenizer_info_new(const char* const* vocab, size_t vocab_size,
                         const int vocab_type,
                         const int32_t* eos_tokens, size_t eos_tokens_size);
void tokenizer_info_free(void* tokenizer_info);

#ifdef __cplusplus
}
#endif
