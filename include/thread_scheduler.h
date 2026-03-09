#pragma once
#include <cstdint>
#include <functional>
#include <vector>
#include <thread>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif

// A minimal, portable thread scheduler used by C student's part
// Provides simple parallel_for over a 1D iteration space.
class ThreadScheduler {
public:
  static ThreadScheduler& instance();

  void set_num_threads(int n);
  int  get_num_threads() const;

  template <typename F>
  inline void parallel_for_1d(int64_t n, F&& fn) {
    int t = num_threads_.load();
    if (t <= 1 || n <= 1) {
      for (int64_t i = 0; i < n; ++i) fn(i);
      return;
    }

#ifdef _OPENMP
    // Use OpenMP when available
    omp_set_num_threads(t);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
      fn(i);
    }
#else
    // Fallback to std::thread
    int64_t chunk = (n + t - 1) / t;
    std::vector<std::thread> threads;
    threads.reserve(t);

    for (int tid = 0; tid < t; ++tid) {
      int64_t start = tid * chunk;
      if (start >= n) break;
      int64_t end = std::min(start + chunk, n);
      threads.emplace_back([start, end, &fn]() {
        for (int64_t i = start; i < end; ++i) fn(i);
      });
    }
    for (auto& th : threads) th.join();
#endif
  }

private:
  ThreadScheduler() : num_threads_(1) {}
  std::atomic<int> num_threads_;
};