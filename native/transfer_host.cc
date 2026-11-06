// The transfer engine on host threads. See host_device.h.
#include "host_device.h"

using Device = lt::HostDevice;

#include "transfer_core.h"

extern "C" {

LT_EXPORT int lt_host_stream(lt_engine* engine, void** out) {
    return lt::guarded(engine, [&] { *out = engine->impl.device().extra_stream(); });
}

LT_EXPORT int lt_host_call(lt_engine* engine, void* stream, lt_host_kernel kernel, void* context) {
    return lt::guarded(engine, [&] { engine->impl.device().call(Device::stream_from_handle(stream), kernel, context); });
}

LT_EXPORT int lt_host_set_paused(lt_engine* engine, int paused) {
    return lt::guarded(engine, [&] { engine->impl.device().set_paused(paused != 0); });
}

LT_EXPORT int lt_host_set_copy_delay_us(lt_engine* engine, uint32_t delay_us) {
    return lt::guarded(engine, [&] { engine->impl.device().set_copy_delay_us(delay_us); });
}

LT_EXPORT int lt_host_set_pin_limit(lt_engine* engine, uint64_t bytes) {
    return lt::guarded(engine, [&] { engine->impl.device().set_pin_limit(bytes); });
}

}  // extern "C"
