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
      int multiStep,
      int numEnvs,
      float gamma,
      float eta,
      int seqLen,
      int numPlayer,
      std::shared_ptr<rela::RNNPrioritizedReplay> replayBuffer)
      : runner_(std::move(runner))
      , numEnvs_(numEnvs)
      , numPlayer_(numPlayer)
      , r2d2Buffer_(std::make_unique<rela::R2D2Buffer>(numEnvs, numPlayer, multiStep, seqLen))
      , multiStepBuffer_(std::make_unique<rela::MultiStepBuffer>(multiStep, numEnvs, gamma))
      , replayBuffer_(std::move(replayBuffer))
      , eta_(eta)
      , hidden_(getH0(numEnvs, numPlayer))
      , numAct_(0) {
    }

  R2D2Actor(std::shared_ptr<rela::BatchRunner> runner, int numPlayer)
      : runner_(std::move(runner))
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

  //virtual rela::TensorDict act(const rela::TensorDict& obs, HanabiEnv& env) {
  virtual rela::TensorDict act(const rela::TensorDict& obs) {
      //std::cout << ":: start c++ act ::" << std::endl;
      torch::NoGradGuard ng;
      assert(!hidden_.empty());

      if (replayBuffer_ != nullptr) {
          historyHidden_.push_back(hidden_);
      }

      // to avoid adding hid into obs;
      auto input = obs;
      for (auto& kv : hidden_) {
          // convert to batch_first
          auto ret = input.emplace(kv.first, kv.second.transpose(0, 1));
          assert(ret.second);
      }

      int slot = -1;
      auto futureReply = runner_->call("act", input, &slot);
      auto reply = futureReply->get(slot);

      for (auto& kv : hidden_) {
          auto newHidIt = reply.find(kv.first);
          assert(newHidIt != reply.end());
          assert(newHidIt->second.sizes() == kv.second.transpose(0, 1).sizes());
          hidden_[kv.first] = newHidIt->second.transpose(0, 1);
          reply.erase(newHidIt);
      }

      // for (auto& kv : reply) {
      //   std::cout << "reply: " << kv.first << ", " << kv.second << std::endl;
      // }

      if (replayBuffer_ != nullptr) {
          multiStepBuffer_->pushObsAndAction(obs, reply);
      }

      numAct_ += numEnvs_;
      return reply;
  }

  // r is float32 tensor, t is byte tensor
  virtual void postAct(const torch::Tensor& r, const torch::Tensor& t) {
      if (replayBuffer_ == nullptr) {
          return;
      }

      // assert(replayBuffer_ != nullptr);
      multiStepBuffer_->pushRewardAndTerminal(r, t);

      // if ith state is terminal, reset hidden states
      // h0: [num_layers * num_directions, batch, hidden_size]
      rela::TensorDict h0 = getH0(1, numPlayer_);
      auto terminal = t.accessor<bool, 1>();
      // std::cout << "terminal size: " << t.sizes() << std::endl;
      // std::cout << "hid size: " << hidden_["h0"].sizes() << std::endl;
      for (int i = 0; i < terminal.size(0); i++) {
          if (!terminal[i]) {
              continue;
          }
          for (auto& kv : hidden_) {
              // [numLayer, numEnvs, hidDim]
              // [numLayer, numEnvs, numPlayer (>1), hidDim]
              kv.second.narrow(1, i * numPlayer_, numPlayer_) = h0.at(kv.first);
          }
      }

      if (replayBuffer_ == nullptr) {
          return;
      }
      assert(multiStepBuffer_->size() == historyHidden_.size());

      if (!multiStepBuffer_->canPop()) {
          assert(!r2d2Buffer_->canPop());
          return;
      }

      {
          rela::FFTransition transition = multiStepBuffer_->popTransition();
          rela::TensorDict hid = historyHidden_.front();
          rela::TensorDict nextHid = historyHidden_.back();
          historyHidden_.pop_front();

          auto input = transition.toDict();
          for (auto& kv : hid) {
              auto ret = input.emplace(kv.first, kv.second.transpose(0, 1));
              assert(ret.second);
          }
          for (auto& kv : nextHid) {
              auto ret = input.emplace("next_" + kv.first, kv.second.transpose(0, 1));
              assert(ret.second);
          }

          int slot = -1;
          auto futureReply = runner_->call("compute_priority", input, &slot);
          auto priority = futureReply->get(slot)["priority"];

          r2d2Buffer_->push(transition, priority, hid);
      }

      if (!r2d2Buffer_->canPop()) {
          return;
      }

      std::vector<rela::RNNTransition> batch;
      torch::Tensor seqBatchPriority;
      torch::Tensor batchLen;

      std::tie(batch, seqBatchPriority, batchLen) = r2d2Buffer_->popTransition();
      auto priority = aggregatePriority(seqBatchPriority, batchLen, eta_);
      replayBuffer_->add(batch, priority);
  }

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

