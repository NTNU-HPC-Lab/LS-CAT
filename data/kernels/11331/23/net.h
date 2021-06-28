#ifndef CAFFE2_CORE_NET_H_
#define CAFFE2_CORE_NET_H_

#include <atomic>
#include <climits>
#include <cstddef>
#include <thread>  // NOLINT
#include <typeinfo>
#include <vector>

#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/simple_queue.h"

namespace caffe2 {

class OperatorBase;

// Net is a thin struct that owns all the operators together with the operator
// contexts.
class NetBase {
 public:
  NetBase(const NetDef& net_def, Workspace* ws) {}
  virtual ~NetBase() {}
  virtual bool Verify() = 0;
  virtual bool Run() = 0;

  DISABLE_COPY_AND_ASSIGN(NetBase);
};

// Essentially, we won't expect too many Net instances, so we will simply
// have a function that produces different net implementations. If needed we can
// switch to a registration pattern later.
NetBase* CreateNet(const NetDef& net_def, Workspace* ws);

// This is the very basic structure you need to run a network - all it
// does is simply to run everything in sequence. If you want more fancy control
// such as a DAG-like execution, check out other better net implementations.
class SimpleNet final : public NetBase {
 public:
  SimpleNet(const NetDef& net_def, Workspace* ws);
  bool Verify() override;
  bool Run() override;

 protected:
  vector<unique_ptr<OperatorBase> > operators_;

  DISABLE_COPY_AND_ASSIGN(SimpleNet);
};

namespace internal {
struct OperatorNode {
  unique_ptr<OperatorBase> operator_;
  vector<int> children_;
  vector<int> parents_;
  std::atomic<int> runtime_parent_count_;
};
}

class ParallelNet final : public NetBase {
 public:
  ParallelNet(const NetDef& net_def, Workspace* ws);
  ~ParallelNet();
  bool Verify() override;
  bool Run() override;
  // WorkerFunction() is a function wrapper to allow us to run worker threads.
  // It checks out one ready-to-run operator from the job queue, runs it,
  // notifies all its children, and for any children that is ready, enqueues
  // it to the job queue.
  void WorkerFunction();

 protected:
  vector<internal::OperatorNode> operator_nodes_;
  vector<int> initial_frontier_;
  SimpleQueue<int> job_queue_;
  std::vector<std::thread> workers_;
  int remaining_ops_;
  bool success_;
  std::mutex remaining_ops_mutex_;
  std::condition_variable cv_;

  DISABLE_COPY_AND_ASSIGN(ParallelNet);
};

}  // namespace caffe2

#endif  // CAFFE2_CORE_NET_H_
