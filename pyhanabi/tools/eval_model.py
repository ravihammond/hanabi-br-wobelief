# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import sys

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)

import numpy as np
import torch
import r2d2
import utils
from eval import evaluate
# from obl_model import obl_model


def evaluate_agents(agents, num_game, seed, bomb, device, num_run=1, verbose=True):
    num_player = len(agents)
    assert num_player > 1, "1 weight file per player"

    scores = []
    perfect = 0
    for i in range(num_run):
        _, _, score, p = evaluate(
            agents,
            num_game,
            num_game * i + seed,
            bomb,
            0,
            True,  # in op paper, sad was a default
            device=device,
        )
        scores.extend(score)
        perfect += p

    mean = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / (num_game * num_run)
    scores_string = ""
    for score in scores:
        scores_string += str(score)
        scores_string += ","
    print(scores_string)
    if verbose:
        print("score: %f +/- %f" % (mean, sem), "; perfect: ", perfect_rate)
    return mean, sem, perfect_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", default="sad", type=str, help="sad/op/obl")
    parser.add_argument("--num_game", default=5000, type=int)
    parser.add_argument(
        "--num_run", default=1, type=int, help="total num game = num_game * num_run"
    )
    # config for model from sad paper
    parser.add_argument("--weight", default="../sad/models/sad/model0.pthw", type=str)
    parser.add_argument("--num_player", default=2, type=int)
    # config for model from op paper
    parser.add_argument(
        "--method", default="sad-aux-op", type=str, help="sad-aux-op/sad-aux/sad-op/sad"
    )
    parser.add_argument("--idx1", default=0, type=int, help="which model to use?")
    parser.add_argument("--idx2", default=0, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)

    args = parser.parse_args()

    if args.paper == "sad":
       # assert os.path.exists(args.weight)
        # we are doing self player, all players use the same weight
       # weight_files = [args.weight for _ in range(args.num_player)]
        assert os.path.exists("exps/exp1/br_234567rand_epoch_800.pthw")
        assert os.path.exists("models/sad/sad_2p_8.pthw")
        assert os.path.exists("model_w_belief_module_br_234567_epoch300.pthw")
        weight_files = ["models/sad/sad_2p_4.pthw", "models/sad/sad_2p_11.pthw"]
        #agents = utils.load_sad_beliefmodule_model(weight_files, args.device)
        agents = utils.load_sad_model(weight_files, args.device)
    elif args.paper == "op":
        agents = utils.load_op_model(args.method, args.idx1, args.idx2, args.device)
    elif args.paper == "obl":
        agents = [obl_model, obl_model]

    # fast evaluation for 5k games
    evaluate_agents(
        agents, args.num_game, np.random.randint(0,20000), 0, num_run=args.num_run, device=args.device
    )
