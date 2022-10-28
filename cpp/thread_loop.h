// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <stdio.h>

#include "cpp/r2d2_actor.h"
#include "rela/r2d2_actor.h"
#include "rela/thread_loop.h"

#define PR true

using HanabiVecEnv = rela::VectorEnv<HanabiEnv>;

class HanabiThreadLoop : public rela::ThreadLoop {
    public:
        HanabiThreadLoop(
                std::shared_ptr<R2D2Actor> actor,
                std::shared_ptr<HanabiVecEnv> vecEnv,
                bool eval)
            : actors_({std::move(actor)})
            , vecEnv_(std::move(vecEnv))
            , eval_(eval) {
                assert(actors_.size() >= 1);
                if (eval_) {
                    assert(vecEnv_->size() == 1);
                }
            }

        HanabiThreadLoop(
                std::vector<std::shared_ptr<R2D2Actor>> actors,
                std::shared_ptr<HanabiVecEnv> vecEnv,
                bool eval)
            : actors_(std::move(actors))
              , vecEnv_(std::move(vecEnv))
              , eval_(eval) {
                  assert(actors_.size() >= 1);
                  if (eval_) {
                      assert(vecEnv_->size() == 1);
                  }
              }

        void mainLoop() final {
            rela::TensorDict obs = {};
            torch::Tensor r;
            torch::Tensor t;
            while (!terminated()) {
                if(PR)printf("\n=======================================\n");

                obs = vecEnv_->reset(obs);

                while (!vecEnv_->anyTerminated()) {
                    auto env = vecEnv_->getEnv(0);
                    if(PR)printf("\n=======================================\n");
                    if(PR)printf("\nScore: %d\n", env->getScore());
                    if(PR)printf("Lives: %d\n", env->getLife());
                    if(PR)printf("Information: %d\n", env->getInfo());
                    auto deck = env->getHleState().Deck();
                    if(PR)printf("Deck: %d\n", deck.Size());
                    std::string colours = "RYGWB";
                    auto fireworks = env->getFireworks();
                    if(PR)printf("Fireworks: ");
                    for (unsigned long i = 0; i < colours.size(); i++)
                        if(PR)printf("%c%d ", colours[i], fireworks[i]);
                    if(PR)printf("\n");
                    auto hands = env->getHleState().Hands();
                    int cp = env->getCurrentPlayer();
                    for(unsigned long i = 0; i < hands.size(); i++) {
                        if(PR)printf("Actor %ld hand:%s\n", i,
                                cp == (int)i ? " <-- current player" : ""); 
                        auto hand = hands[i].ToString();
                        hand.pop_back();
                        if(PR)printf("%s\n", hand.c_str());
                    }
                    if(PR)printf("\n----\n");

                    if (terminated()) {
                        break;
                    }

                    if (paused()) {
                        waitUntilResume();
                    }

                    rela::TensorDict reply;
                    std::vector<rela::TensorDict> replyVec;
                    for (int i = 0; i < (int)actors_.size(); ++i) {
                        auto input = rela::tensor_dict::narrow(obs, 1, i, 1, true);
                        env = vecEnv_->getEnv(i / 2);
                        int curPlayer = env->getCurrentPlayer();
                        if(PR)printf("\n[player %d acting]%s\n", i % 2, 
                                curPlayer == (int)(i % 2) ? " <-- current player" : "");
                        auto rep = actors_[i]->act(input, env);
                        replyVec.push_back(rep);
                    }
                    reply = rela::tensor_dict::stack(replyVec, 1);
                    std::tie(obs, r, t) = vecEnv_->step(reply);

                    if(PR)printf("\n----\n");

                    if (eval_) {
                        continue;
                    }

                    for (int i = 0; i < (int)actors_.size(); ++i) {
                        env = vecEnv_->getEnv(i / 2);
                        int curPlayer = env->getCurrentPlayer();
                        if(PR)printf("\n[player %d postAct]%s\n", i % 2, 
                                curPlayer == (int)(i % 2) ? " <-- current player" : "");
                        actors_[i]->postAct(r, t);
                    }
                }

                // eval only runs for one game
                if (eval_) {
                    break;
                }
            }
        }

    private:
        std::vector<std::shared_ptr<R2D2Actor>> actors_;
        std::shared_ptr<HanabiVecEnv> vecEnv_;
        const bool eval_;
};
