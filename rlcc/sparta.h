#pragma once

#include "rlcc/game_sim.h"
#include "rlcc/hybrid_model.h"
#include "rlcc/hand_dist.h"
#include "rela/tensor_dict.h"

namespace sparta {

class SpartaActor {

 public:
  SpartaActor(
      int index,
      std::shared_ptr<rela::BatchRunner> bpRunner,
      int seed)
      : index(index)
      , rng_(seed)
      , prevModel_(index)
      , model_(index) {
    assert(bpRunner != nullptr);
    model_.setBpModel(bpRunner, getH0(*bpRunner, 1));
  }

  void setPartners(std::vector<std::shared_ptr<SpartaActor>> partners, py::object pyModel) {
    partners_ = std::move(partners);
 //   partners_[0]->model_.setBpModelPy(pyModel);
  }

  void changeBpPy(py::object pyModel) {
    model_.setBpModelPy(pyModel);
  }

  void changeBp(std::shared_ptr<rela::BatchRunner> bpRunner) {
	model_.setBpModel(bpRunner, getH0(*bpRunner, 1));
  }

  // void updateBelief(const approx_search::GameSimulator& env, int numThread) {
  //   assert(callOrder_ == 0);
  //   ++callOrder_;

  //   const auto& state = env.state();
  //   int curPlayer = state.CurPlayer();
  //   int numPlayer = env.game().NumPlayers();
  //   assert((int)partners_.size() == numPlayer);
  //   int prevPlayer = (curPlayer - 1 + numPlayer) % numPlayer;
  //   std::cout << "prev player: " << prevPlayer << std::endl;

  //   auto [obs, lastMove, cardCount, myHand] =
  //       observeForSearch(env.state(), index, hideAction, false);

  //   approx_search::updateBelief(
  //       prevState_,
  //       env.game(),
  //       lastMove,
  //       cardCount,
  //       myHand,
  //       partners_[prevPlayer]->prevModel_,
  //       index,
  //       handDist_,
  //       numThread);
  // }

  std::tuple<rela::TensorDict, std::vector<int>> updateBelief(const approx_search::GameSimulator& env, int numThread) {
    assert(callOrder_ == 0);
    ++callOrder_;

    const auto& state = env.state();
    int curPlayer = state.CurPlayer();
    int numPlayer = env.game().NumPlayers();
    assert((int)partners_.size() == numPlayer);
    int prevPlayer = (curPlayer - 1 + numPlayer) % numPlayer;
    std::cout << "prev player: " << prevPlayer << std::endl;

    auto [obss, lastMove, cardCount, myHand] =
        observeForSearch(env.state(), index, hideAction, false);
   // approx_search::updateBelief(
   //   prevState_,
   //   env.game(),
   //   lastMove,
   //   cardCount,
   //   myHand,
   //   partners_[prevPlayer]->prevModel_,
   //   index,
   //   handDist_,
   //   numThread);

    bool shuffleColor = false;
    const std::vector<int>& colorPermute = std::vector<int>();
    const std::vector<int>& invColorPermute = std::vector<int>();
    bool trinary = true;
    bool sad = true;

    const auto& game = *(env.state().ParentGame());
    auto obs = hle::HanabiObservation(env.state(), index, true);
    auto encoder = hle::CanonicalObservationEncoder(&game);

    std::vector<float> vS = encoder.Encode(
      obs,
      true,  // regardless of the flag, splitPrivatePulic/convertSad will mask out this
             // field
      std::vector<int>(),  // shuffle card
      shuffleColor,
      colorPermute,
      invColorPermute,
      hideAction);

    rela::TensorDict feat;
    if (!sad) {
      feat = splitPrivatePublic(vS, game);
    } else {
      // only for evaluation
      auto vA =
          encoder.EncodeLastAction(obs, std::vector<int>(), shuffleColor, colorPermute);
      feat = convertSad(vS, vA, game);
    }

    return {feat, cardCount};//observe(env.state(), index, false, std::vector<int>(), std::vector<int>(), hideAction, true, true);
  }

  void observe(const approx_search::GameSimulator& env) {
    // assert(callOrder_ == 1);
    ++callOrder_;

    const auto& state = env.state();
    model_.observeBeforeAct(env, 0);

    if (prevState_ == nullptr) {
      prevState_ = std::make_unique<hle::HanabiState>(state);
    } else {
      *prevState_ = state;
    }
  }

  int decideAction(const approx_search::GameSimulator& env) {
    // assert(callOrder_ == 2);
    callOrder_ = 0;

    prevModel_ = model_;  // this line can only be in decide action
    return model_.decideAction(env, false);
  }

  // should be called after decideAction?
  hle::HanabiMove spartaSearch(
      const approx_search::GameSimulator& env,
      hle::HanabiMove bpMove,
      int numSearch,
      float threshold,
      std::vector<int> cardCount,
      std::vector<int> samples1,
      std::vector<int> samples2,
      std::vector<int> samples3,
      std::vector<int> samples4,
      std::vector<int> samples5);

  const int index;
  const bool hideAction = false;

 private:
  mutable std::mt19937 rng_;

  approx_search::HybridModel prevModel_;
  approx_search::HybridModel model_;
  approx_search::HandDistribution handDist_;

  std::vector<std::shared_ptr<SpartaActor>> partners_;
  std::unique_ptr<hle::HanabiState> prevState_ = nullptr;

  int callOrder_ = 0;
};

}  // namespace approx_search
