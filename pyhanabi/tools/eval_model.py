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
import newr2d2
import utils
from eval import evaluate
#from obl_model import obl_model
import torch.onnx
# import netron
import copy

def load_sad_model(weight_files, device):
    agents = []
    for weight_file in weight_files:
        if "sad" in weight_file or "aux" in weight_file:
            sad = True
        else:
            sad = False

        state_dict = torch.load(weight_file, map_location=device)
        input_dim = state_dict["net.0.weight"].size()[1]
        hid_dim = 512
        output_dim = state_dict["fc_a.weight"].size()[0]

        agent = r2d2.R2D2Agent(
            False, 3, 0.999, 0.9, device, input_dim, hid_dim, output_dim, 2, 5, False
        ).to(device)
        utils.load_weight(agent.online_net, weight_file, device)
        agents.append(agent)
    return agents


def load_op_model(method, idx1, idx2, device):
    """load op models, op models was trained only for 2 player
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # assume model saved in root/models/op
    folder = os.path.join(root, "models", "op", method)
    print(folder)
    #/home/zhaoogroup/code/tao_huang/hanabi/models/op/sad-aux-op
    #/home/zhaoogroup/code/tao_huang/hanabi/models/op/sad-aux-op/M0.pthw
    agents = []
    for idx in [idx1, idx2]:
        if idx >= 0 and idx < 3:
            num_fc = 1
            skip_connect = False
        elif idx >= 3 and idx < 6:
            num_fc = 1
            skip_connect = True
        elif idx >= 6 and idx < 9:
            num_fc = 2
            skip_connect = False
        else:
            num_fc = 2
            skip_connect = True
        weight_file = os.path.join(folder, f"M{idx}.pthw")
        if not os.path.exists(weight_file):
            print(f"Cannot find weight at: {weight_file}")
            assert False

        state_dict = torch.load(weight_file)
        input_dim = state_dict["net.0.weight"].size()[1]
        hid_dim = 512
        output_dim = state_dict["fc_a.weight"].size()[0]
        agent = r2d2.R2D2Agent(
            False,
            3,
            0.999,
            0.9,
            device,
            input_dim,
            hid_dim,
            output_dim,
            2,
            5,
            False,
            num_fc_layer=num_fc,
            skip_connect=skip_connect,
        ).to(device)
        utils.load_weight(agent.online_net, weight_file, device)
        agents.append(agent)
    return agents

def load_dict_model(method, idx1, idx2, device):
    """load op models, op models was trained only for 2 player
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # assume model saved in root/models/op
    folder = os.path.join(root, "models", "op", method)
    agents = []
    for idx in [idx1, idx2]:
        if idx >= 0 and idx < 3:
            num_fc = 1
            skip_connect = False
        elif idx >= 3 and idx < 6:
            num_fc = 1
            skip_connect = True
        elif idx >= 6 and idx < 9:
            num_fc = 2
            skip_connect = False
        else:
            num_fc = 2
            skip_connect = True
        weight_file = os.path.join(folder, f"M{idx}.pthw")
        if not os.path.exists(weight_file):
            print(f"Cannot find weight at: {weight_file}")
            assert False

        state_dict = torch.load(weight_file)
        input_dim = state_dict["net.0.weight"].size()[1]
        hid_dim = 512
        output_dim = state_dict["fc_a.weight"].size()[0]
        agent = r2d2.SadPlayer2New(
            False,
            3,
            0.999,
            0.9,
            device,
            input_dim,
            hid_dim,
            output_dim,
            2,
            5,
            False,
            num_fc_layer=num_fc,
            skip_connect=skip_connect,
        ).to(device)
        # print("net: ",agent.net)
        utils.load_weight(agent.online_net, weight_file, device)
        agents.append(agent)
    return agents

def loadBP(method,device):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # assume model saved in root/models/op
    folder = os.path.join(root, "models", "op", method)
    agents = []
    idx=1

    num_fc = 1
    skip_connect = False

    weight_file = os.path.join(folder, f"M{idx}.pthw")
    if not os.path.exists(weight_file):
        print(f"Cannot find weight at: {weight_file}")
        assert False

    state_dict = torch.load(weight_file)
    for k,v in state_dict.items():
        print(str(k),v.size())
    input_dim = state_dict["net.0.weight"].size()[1]
    hid_dim = 512
    output_dim = state_dict["fc_a.weight"].size()[0]
    print("loading model: ",input_dim,output_dim)
    agent = r2d2.R2D2Agent(
        True,
        3,
        0.999,
        0.9,
        device,
        input_dim,
        hid_dim,
        output_dim,
        2,
        5,
        True,
        num_fc_layer=num_fc,
        skip_connect=skip_connect,
    ).to(device)
    utils.load_weight(agent.online_net, weight_file, device)
    return agent

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
    parser.add_argument("--weight", default=None, type=str)
    parser.add_argument("--num_player", default=None, type=int)
    # config for model from op paper
    parser.add_argument(
        "--method", default="sad-aux-op", type=str, help="sad-aux-op/sad-aux/sad-op/sad"
    )
    parser.add_argument("--idx1", default=1, type=int, help="which model to use?")
    parser.add_argument("--idx2", default=1, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)

    args = parser.parse_args()

    if args.paper == "sad":
        assert os.path.exists(args.weight)
        # we are doing self player, all players use the same weight
        weight_files = [args.weight for _ in range(args.num_player)]
        agents = load_sad_model(weight_files, args.device)
        m = torch.jit.script(agents[0])
        torch.jit.save(m, 'new_sad.pth')
        print("finish")
    elif args.paper == "op":
        agents = load_op_model(args.method, args.idx1, args.idx2, args.device)
        m = torch.jit.script(agents[0])
        torch.jit.save(m, 'new_op.pth')
        # print(torch.jit.load("new_oppth"))
        print("finish")
    elif args.paper == "obl":
        agents = [obl_model, obl_model]

    elif args.paper == "load":
        agents=load_dict_model(args.method, args.idx1, args.idx2, args.device)
        # print(agents[0].state_dict().keys())

        # agent=agents[0]
        # for k in agent.state_dict().keys():
        #     if k.startswith("target"):
        #         agent.state_dict().pop(k)
        # print(agent.state_dict().keys())


        m = torch.jit.script(agents[0])
        # torch.jit.save({
        #     "net":m.state_dict()["net"]
        # }, 'jit_op.pth')
        torch.jit.save(m, 'final_new_r2d2.pth')
        print(torch.jit.load("final_new_r2d2.pth"))

        # python tools/eval_model.py --paper op --method sad-aux-op --idx1 0 --idx2 0

    # fast evaluation for 5k games
    # evaluate_agents(
    #     agents, args.num_game, 1, 0, num_run=args.num_run, device=args.device
    # )
