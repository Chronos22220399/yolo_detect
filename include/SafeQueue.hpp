#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T> class SafeQueue {
public:
  SafeQueue(size_t capacity = 10) : capacity_(capacity) {}

  void push(const T &value) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.size() >= capacity_) {
      queue_.pop(); // 丢弃最旧帧，保持实时性
    }
    queue_.push(value);
    cond_.notify_one();
  }

  bool pop(T &value) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return !queue_.empty(); });
    value = queue_.front();
    queue_.pop();
    return true;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable cond_;
  std::queue<T> queue_;
  size_t capacity_;
};
