#include "include/cxgrammar/error_handler.h"

static error_handler_closure _error_handler = nullptr;

extern "C" {

void set_error_handler(error_handler_closure handler) {
    _error_handler = handler;
}

void catch_error(const char* error_message) {
    if (_error_handler) {
        _error_handler(error_message);
    }
}

}
