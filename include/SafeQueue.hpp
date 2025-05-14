#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T> class SafeQueue {
public:
  SafeQueue(size_t capacity = 10) : capacity_(capacity) {}

  bool try_pop(T &value) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty())
      return false;

    value = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  bool try_push(const T value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.size() >= capacity_) {
      queue_.pop();
    }
    queue_.push(value);
    cond_.notify_one();
    return true;
  }

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

  size_t size() const { return queue_.size(); }

private:
  mutable std::mutex mutex_;
  std::condition_variable cond_;
  std::queue<T> queue_;
  size_t capacity_;
};
