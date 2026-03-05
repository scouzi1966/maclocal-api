#pragma once
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void* grammar_matcher_new(void* compiled_grammar);
void  grammar_matcher_fill_next_token_bitmask(void* matcher, void* bitmask);
bool  grammar_matcher_accept_token(void* matcher, int32_t token_id);
bool  grammar_matcher_is_terminated(void* matcher);
void  grammar_matcher_reset(void* matcher);
void  grammar_matcher_free(void* matcher);

/// Returns the number of int32 elements needed for a bitmask covering vocab_size tokens.
int grammar_bitmask_size(int vocab_size);

#ifdef __cplusplus
}
#endif
