// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once

#include "rela/batch_runner.h"
#include "rela/transition_buffer.h"
#include "rela/prioritized_replay.h"
#include "cpp/hanabi_env.h"

torch::Tensor aggregatePriority(torch::Tensor priority, torch::Tensor seqLen, float eta);

class R2D2Actor {
  public:
    R2D2Actor(
      std::shared_ptr<rela::BatchRunner> runner,
      int playerIdx,
      int multiStep,
      int numEnvs,
      float gamma,
      float eta,
      int seqLen,
      int numPlayer,
      std::shared_ptr<rela::RNNPrioritizedReplay> replayBuffer)
      : runner_(std::move(runner))
      , playerIdx_(playerIdx)
      , numEnvs_(numEnvs)
      , numPlayer_(numPlayer)
      , r2d2Buffer_(std::make_unique<rela::R2D2Buffer>(numEnvs, numPlayer, multiStep, seqLen))
      , multiStepBuffer_(std::make_unique<rela::MultiStepBuffer>(multiStep, numEnvs, gamma))
      , replayBuffer_(std::move(replayBuffer))
      , eta_(eta)
      , hidden_(getH0(numEnvs, numPlayer))
      , numAct_(0) {
    }

  R2D2Actor(
          std::shared_ptr<rela::BatchRunner> runner, 
          int playerIdx,
          int numPlayer)
      : runner_(std::move(runner))
      , playerIdx_(playerIdx)
      , numEnvs_(1)
      , numPlayer_(numPlayer)
      , r2d2Buffer_(nullptr)
      , multiStepBuffer_(nullptr)
      , replayBuffer_(nullptr)
      , eta_(0)
      , hidden_(getH0(1, numPlayer))
      , numAct_(0) {
  }

  int numAct() const {
    return numAct_;
  }

  virtual rela::TensorDict act(const rela::TensorDict& obs, 
          std::shared_ptr<HanabiEnv> env);

  // r is float32 tensor, t is byte tensor
  virtual void postAct(const torch::Tensor& r, const torch::Tensor& t);

  protected:
  rela::TensorDict getH0(int numEnvs, int numPlayer) {
      std::vector<torch::jit::IValue> input{numEnvs * numPlayer};
      auto model = runner_->jitModel();
      auto output = model.get_method("get_h0")(input);
      auto h0 = rela::tensor_dict::fromIValue(output, torch::kCPU, true);
      // for (auto& kv : h0) {
      //   h0[kv.first] = kv.second.transpose(0, 1);
      // }
      return h0;
  }

  std::shared_ptr<rela::BatchRunner> runner_;
  const int playerIdx_;
  const int numEnvs_;
  const int numPlayer_;

  std::deque<rela::TensorDict> historyHidden_;
  std::unique_ptr<rela::R2D2Buffer> r2d2Buffer_;
  std::unique_ptr<rela::MultiStepBuffer> multiStepBuffer_;
  std::shared_ptr<rela::RNNPrioritizedReplay> replayBuffer_;

  const float eta_;

  rela::TensorDict hidden_;
  std::atomic<int> numAct_;
};

