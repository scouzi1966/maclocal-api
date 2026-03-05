#pragma once
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void* compile_json_schema_grammar(void* tokenizer_info,
                                  const char* schema, size_t length,
                                  int indent);
void* compile_ebnf_grammar(void* tokenizer_info,
                            const char* grammar, size_t length);
void* compile_regex_grammar(void* tokenizer_info,
                             const char* regex, size_t length);
void compiled_grammar_free(void* compiled_grammar);

#ifdef __cplusplus
}
#endif
