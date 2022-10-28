#include "cpp/r2d2_actor.h"

class ConventionActor: public R2D2Actor {
    ConventionActor(
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

    ConventionActor(
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

    private:
};
