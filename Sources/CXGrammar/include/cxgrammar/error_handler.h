#pragma once
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*error_handler_closure)(const char*);
void set_error_handler(error_handler_closure handler);
void catch_error(const char* error_message);

#ifdef __cplusplus
}
#endif
