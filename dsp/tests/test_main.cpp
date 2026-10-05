#include <chrono>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <new>

#include "test_framework.h"

namespace vt {

std::atomic<long> g_allocations{0};
static int g_failures = 0;
static const char* g_current = "";

std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

void fail(const char* file, int line, const std::string& msg) {
    ++g_failures;
    std::printf("    FAIL %s:%d [%s] %s\n", file, line, g_current, msg.c_str());
}

void note(const char* fmt, ...) {
    std::printf("    | ");
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::printf("\n");
}

}  // namespace vt

// Global allocation counter: lets tests prove that Engine::process() never
// touches the heap (a hard requirement for real-time audio callbacks).
void* operator new(std::size_t n) {
    vt::g_allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) {
    vt::g_allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int run = 0, failedTests = 0;
    for (const vt::TestCase& t : vt::registry()) {
        if (filter && std::strstr(t.name, filter) == nullptr) continue;
        vt::g_current = t.name;
        const int before = vt::g_failures;
        std::printf("[ RUN  ] %s\n", t.name);
        std::fflush(stdout);
        const auto t0 = std::chrono::steady_clock::now();
        t.fn();
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const bool ok = vt::g_failures == before;
        std::printf("[ %s ] %s (%.0f ms)\n", ok ? " OK " : "FAIL", t.name, ms);
        ++run;
        if (!ok) ++failedTests;
    }
    std::printf("\n%d tests, %d failed, %d failed checks\n", run, failedTests, vt::g_failures);
    return failedTests == 0 ? 0 : 1;
}
