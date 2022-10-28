#include <stdio.h>
#include <iostream>

#include "cpp/r2d2_actor.h"

#define PR true

using namespace std;

torch::Tensor aggregatePriority(torch::Tensor priority, torch::Tensor seqLen, float eta) {
    assert(priority.device().is_cpu() && seqLen.device().is_cpu());
    auto mask = torch::arange(0, priority.size(0));
    mask = (mask.unsqueeze(1) < seqLen.unsqueeze(0)).to(torch::kFloat32);
    assert(priority.sizes() == mask.sizes());
    priority = priority * mask;

    auto pMean = priority.sum(0) / seqLen;
    auto pMax = std::get<0>(priority.max(0));
    auto aggPriority = eta * pMax + (1.0 - eta) * pMean;
    return aggPriority.detach();
}

rela::TensorDict R2D2Actor::act(const rela::TensorDict& obs, 
        std::shared_ptr<HanabiEnv> env) {
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

    //for (auto& it: reply) {
        //cout << it.first << endl;
        //cout << it.second << endl;
    //}

    int action = reply["a"].item<int64_t>();
    printf("action: %d\n", action);
    auto& state = env->getHleState();
    printf("actionx: %d\n", action);
    auto move = state.ParentGame()->GetMove(action);
    if(PR)printf("Playing move: %s\n", move.ToString().c_str());

    return reply;
}

// r is float32 tensor, t is byte tensor
void R2D2Actor::postAct(const torch::Tensor& r, const torch::Tensor& t) {
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
