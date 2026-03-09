// Simple thread scheduler for parallelizing row-wise loops in SIMD ops
#include <vector>
#include <thread>
#include <algorithm>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "thread_scheduler.h"

ThreadScheduler& ThreadScheduler::instance() {
  static ThreadScheduler inst;
  return inst;
}

void ThreadScheduler::set_num_threads(int n) {
  if (n < 1) n = 1;
  num_threads_.store(n);
#ifdef _OPENMP
  omp_set_num_threads(n);
#endif
}

int ThreadScheduler::get_num_threads() const {
  return num_threads_.load();
}