// Minimal self-contained test framework (no external downloads needed).
#pragma once

#include <atomic>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace vt {

struct TestCase {
    const char* name;
    void (*fn)();
};

std::vector<TestCase>& registry();
void fail(const char* file, int line, const std::string& msg);
void note(const char* fmt, ...);
extern std::atomic<long> g_allocations;

struct Registrar {
    Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

}  // namespace vt

#define TEST(name)                                       \
    static void name();                                  \
    static vt::Registrar registrar_##name(#name, name);  \
    static void name()

#define CHECK(cond)                                         \
    do {                                                    \
        if (!(cond)) vt::fail(__FILE__, __LINE__, #cond);   \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                          \
    do {                                                                                               \
        const double va_ = (a), vb_ = (b);                                                             \
        if (!(std::fabs(va_ - vb_) <= (tol))) {                                                        \
            char buf_[256];                                                                            \
            std::snprintf(buf_, sizeof(buf_), "%s = %.4f, expected %.4f +- %.4f", #a, va_, vb_,        \
                          static_cast<double>(tol));                                                   \
            vt::fail(__FILE__, __LINE__, buf_);                                                        \
        }                                                                                              \
    } while (0)

#define CHECK_LE(a, b)                                                                              \
    do {                                                                                            \
        const double va_ = (a), vb_ = (b);                                                          \
        if (!(va_ <= vb_)) {                                                                        \
            char buf_[256];                                                                         \
            std::snprintf(buf_, sizeof(buf_), "%s = %.4f, expected <= %.4f", #a, va_, vb_);         \
            vt::fail(__FILE__, __LINE__, buf_);                                                     \
        }                                                                                           \
    } while (0)

#define CHECK_GE(a, b)                                                                              \
    do {                                                                                            \
        const double va_ = (a), vb_ = (b);                                                          \
        if (!(va_ >= vb_)) {                                                                        \
            char buf_[256];                                                                         \
            std::snprintf(buf_, sizeof(buf_), "%s = %.4f, expected >= %.4f", #a, va_, vb_);         \
            vt::fail(__FILE__, __LINE__, buf_);                                                     \
        }                                                                                           \
    } while (0)
