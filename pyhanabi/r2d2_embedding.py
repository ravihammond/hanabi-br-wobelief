# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from typing import Tuple, Dict
import common_utils

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class R2D2Net(torch.jit.ScriptModule):
    __constants__ = [
        "hid_dim",
        "out_dim",
        "num_lstm_layer",
        "hand_size",
        "skip_connect",
    ]

    def __init__(
        self,
        device,
        in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        hand_size,
        num_fc_layer,
        skip_connect,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_fc_layer = num_fc_layer
        self.num_lstm_layer = num_lstm_layer
        self.hand_size = hand_size
        self.skip_connect = skip_connect

        ff_layers = [nn.Linear(self.in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_fc_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim, self.hid_dim, num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        # for aux task
        self.pred = nn.Linear(self.hid_dim, self.hand_size * 3)

        self.look_up = nn.Embedding(206, 128)
        self.features_to_states = nn.Linear(15*128, 838)
        self.norm = Norm(838)
        self.dropout = nn.Dropout(0.1)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self, priv_s: torch.Tensor, hid: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert priv_s.dim() == 2, "dim should be 2, [batch, dim], get %d" % s.dim()

        priv_s = priv_s.unsqueeze(0)

        reshape = priv_s.shape
        reshape[-1] = 15*128

        x = self.look_up(priv_s.to(torch.long))
        x = self.features_to_states(x.reshape(reshape))
        x = self.norm(x)

        x = self.net(x)
        o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        if self.skip_connect:
            o = o + x
        a = self.fc_a(o)
        a = a.squeeze(0)
        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            one_step = True

        reshape = priv_s.shape
        reshape[-1] = 15*128

        x = self.look_up(priv_s.to(torch.long))
        x = self.features_to_states(x.reshape(reshape))
        x = self.norm(x)

        x = self.net(x)
        if len(hid) == 0:
            o, (h, c) = self.lstm(x)
        else:
            o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))

        a = self.fc_a(o)
        v = self.fc_v(o)
        q = self._duel(v, a, legal_move)

        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)

        return qa, greedy_action, q, o

    @torch.jit.script_method
    def _duel(
        self, v: torch.Tensor, a: torch.Tensor, legal_move: torch.Tensor
    ) -> torch.Tensor:
        assert a.size() == legal_move.size()
        legal_a = a * legal_move
        q = v + legal_a - legal_a.mean(2, keepdim=True)
        return q

    def cross_entropy(self, net, lstm_o, target_p, hand_slot_mask, seq_len):
        # target_p: [seq_len, batch, num_player, 5, 3]
        # hand_slot_mask: [seq_len, batch, num_player, 5]
        logit = net(lstm_o).view(target_p.size())
        q = nn.functional.softmax(logit, -1)
        logq = nn.functional.log_softmax(logit, -1)
        plogq = (target_p * logq).sum(-1)
        xent = -(plogq * hand_slot_mask).sum(-1) / hand_slot_mask.sum(-1).clamp(
            min=1e-6
        )

        if xent.dim() == 3:
            # [seq, batch, num_player]
            xent = xent.mean(2)

        # save before sum out
        seq_xent = xent
        xent = xent.sum(0)
        assert xent.size() == seq_len.size()
        avg_xent = (xent / seq_len).mean().item()
        return xent, avg_xent, q, seq_xent.detach()

    def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
        return self.cross_entropy(self.pred, lstm_o, target, hand_slot_mask, seq_len)


class R2D2Agent(torch.jit.ScriptModule):
    __constants__ = ["vdn", "multi_step", "gamma", "eta", "uniform_priority"]

    def __init__(
        self,
        vdn,
        multi_step,
        gamma,
        eta,
        device,
        in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        hand_size,
        uniform_priority,
        *,
        num_fc_layer=1,
        skip_connect=False,
    ):
        super().__init__()
        self.online_net = R2D2Net(
            device,
            in_dim,
            hid_dim,
            out_dim,
            num_lstm_layer,
            hand_size,
            num_fc_layer,
            skip_connect,
        ).to(device)
        self.target_net = R2D2Net(
            device,
            in_dim,
            hid_dim,
            out_dim,
            num_lstm_layer,
            hand_size,
            num_fc_layer,
            skip_connect,
        ).to(device)
        self.vdn = vdn
        self.multi_step = multi_step
        self.gamma = gamma
        self.eta = eta
        self.uniform_priority = uniform_priority

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        return self.online_net.get_h0(batchsize)

    def clone(self, device, overwrite=None):
        if overwrite is None:
            overwrite = {}
        cloned = type(self)(
            overwrite.get("vdn", self.vdn),
            self.multi_step,
            self.gamma,
            self.eta,
            device,
            self.online_net.in_dim,
            self.online_net.hid_dim,
            self.online_net.out_dim,
            self.online_net.num_lstm_layer,
            self.online_net.hand_size,
            self.uniform_priority,
            num_fc_layer=self.online_net.num_fc_layer,
            skip_connect=self.online_net.skip_connect,
        )
        cloned.load_state_dict(self.state_dict())
        return cloned.to(device)

    def sync_target_with_online(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.jit.script_method
    def greedy_act(
        self,
        priv_s: torch.Tensor,
        legal_move: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        adv, new_hid = self.online_net.act(priv_s, hid)
        legal_adv = (1 + adv - adv.min()) * legal_move
        greedy_action = legal_adv.argmax(1).detach()
        return greedy_action, new_hid

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape
            [batchsize] or [batchsize, num_player]
        """
        obsize, ibsize, num_player = 0, 0, 0
        if self.vdn:
            obsize, ibsize, num_player = obs["priv_s"].size()[:3]
            ### here we go
            batch_size = obs["priv_s"].size(1)

            # 0-25 is partner's cards
                    #   0-25 is first card
                    #   26-51 is second card
                    #   52-77 is third card
                    #   78-103 is fourth card
                    #   104-129 is fifth card
            
            partner_cards = obs["priv_s"][:,:,:,125:250].reshape(obs["priv_s"].size(0),batch_size,2,5,25)
            partner_cards_empty_mask = (partner_cards.sum(-1, keepdim=True) == 0.0)
            partner_cards = torch.cat([partner_cards, partner_cards_empty_mask.float()], -1)
            partner_cards = partner_cards.argmax(dim = 4)

            # 26-66 is remaining deck size

            decksizes = 26 + torch.sum(obs["priv_s"][:,:,:,252:292], -1, dtype = torch.long)

            # 67+5*c-72+5*c is fireworks of colour c

            fireworks = obs["priv_s"][:,:,:,292:317].reshape(obs["priv_s"].size(0),batch_size,2,5,5)
            fireworks_empty_mask = (fireworks.sum(-1, keepdim=True) == 0.0)
            fireworks = torch.cat([fireworks, fireworks_empty_mask.float()], -1)
            fireworks = fireworks.argmax(dim = 4)

            for c in range(5):
                fireworks[:,:,:,c] = 67+5*c+fireworks[:,:,:,c]

            # 93-101 is info tokens

            info_tokens = 93 + torch.sum(obs["priv_s"][:,:,:,317:325], -1, dtype = torch.long)

            # 102-105 is life tokens

            life_tokens = 102 + torch.sum(obs["priv_s"][:,:,:,325:328], -1, dtype = torch.long)

            #

            if torch.sum(obs["priv_s"][1:,:,:,378:431]).item() == 0:
                move_type = torch.ones(obs["priv_s"].size(0), obs["priv_s"].size(1), obs["priv_s"].size(2), dtype = torch.long, device="cuda:1") * 203
                move_affect = torch.ones(obs["priv_s"].size(0), obs["priv_s"].size(1), obs["priv_s"].size(2), dtype = torch.long, device="cuda:1") * 204

            else:
                move_type = obs["priv_s"][1:,:,:,380:384]
                move_type_empty_mask = (move_type.sum(-1, keepdim=True) == 0.0)
                move_type = torch.cat([move_type, move_type_empty_mask.float()], -1)
                move_type = move_type.argmax(dim = 3)
                move_type = 5*move_type + 106

                which_colour = obs["priv_s"][1:,:,:,386:391]
                which_rank = obs["priv_s"][1:,:,:,391:396]
                which_play_disc = obs["priv_s"][1:,:,:,401:406]

                which_colour_empty_mask = (which_colour.sum(-1, keepdim=True) == 0.0)
                which_colour = torch.cat([which_colour_empty_mask.float(), which_colour], -1)

                which_rank_empty_mask = (which_rank.sum(-1, keepdim=True) == 0.0)
                which_rank = torch.cat([which_rank_empty_mask.float(), which_rank], -1)

                which_play_disc_empty_mask = (which_play_disc.sum(-1, keepdim=True) == 0.0)
                which_play_disc = torch.cat([which_play_disc_empty_mask.float(), which_play_disc], -1)

                which_colour = which_colour.argmax(dim = 3)
                which_rank = which_rank.argmax(dim = 3)
                which_play_disc = which_play_disc.argmax(dim = 3)

                move_type += (which_colour + which_rank + which_play_disc - 1)

                which_player = obs["priv_s"][1:,:,:,378:380]
                which_player_empty_mask = (which_player.sum(-1, keepdim=True) == 0.0)
                which_player = torch.cat([which_player, which_player_empty_mask.float()], -1)
                which_player = which_player.argmax(dim = 3)

                move_type += (20*which_player)

                move_affect = obs["priv_s"][1:,:,:,406:431]
                move_affect_empty_mask = (move_affect.sum(-1, keepdim=True) == 0.0)
                move_affect = torch.cat([move_affect, move_affect_empty_mask.float()], -1)
                move_affect = move_affect.argmax(dim = 3)

                move_affect += 146

                move_affect += (obs["priv_s"][1:,:,:,396:401].matmul(2**torch.arange(5, dtype=torch.float, device="cuda:1").flip(0).view(5,1))).reshape(-1, batch_size, 2).to(torch.long)

                move_type = torch.cat([torch.tensor([203 for _ in range(batch_size*2)], device="cuda:1", dtype=torch.long).reshape(1,batch_size,2), move_type], 0)
                move_affect = torch.cat([torch.tensor([204 for _ in range(batch_size*2)], device="cuda:1", dtype=torch.long).reshape(1,batch_size,2), move_affect.to(torch.long)], 0)

            stacked = torch.stack([partner_cards[:,:,:,0], partner_cards[:,:,:,1], partner_cards[:,:,:,2], 
                                    partner_cards[:,:,:,3], partner_cards[:,:,:,4], decksizes, fireworks[:,:,:,0],
                                    fireworks[:,:,:,1], fireworks[:,:,:,2], fireworks[:,:,:,3], fireworks[:,:,:,4], 
                                    info_tokens, life_tokens, move_type, move_affect] , dim=3)

            interleaved = torch.flatten(stacked, start_dim = 0, end_dim = 2)

            # for j in range(batch_size):
            #     for k in range(obs["priv_s"].size(0)):
            #         if torch.sum(obs["priv_s"][k, j, 0, :]) == 0:
            #             interleaved[j, :, k*15:] = 205
            #             break

            ### ouch
            priv_s = interleaved.reshape(-1, 15)
            # priv_s = obs["priv_s"].flatten(0, 2)
            legal_move = obs["legal_move"].flatten(0, 2)
            eps = obs["eps"].flatten(0, 2)
        else:
            obsize, ibsize = obs["priv_s"].size()[:2]
            num_player = 1
            ### here we go
            batch_size = obs["priv_s"].size(1)

            # 0-25 is partner's cards
                    #   0-25 is first card
                    #   26-51 is second card
                    #   52-77 is third card
                    #   78-103 is fourth card
                    #   104-129 is fifth card
            
            partner_cards = obs["priv_s"][:,:,:,125:250].reshape(obs["priv_s"].size(0),batch_size,2,5,25)
            partner_cards_empty_mask = (partner_cards.sum(-1, keepdim=True) == 0.0)
            partner_cards = torch.cat([partner_cards, partner_cards_empty_mask.float()], -1)
            partner_cards = partner_cards.argmax(dim = 4)

            # 26-66 is remaining deck size

            decksizes = 26 + torch.sum(obs["priv_s"][:,:,:,252:292], -1, dtype = torch.long)

            # 67+5*c-72+5*c is fireworks of colour c

            fireworks = obs["priv_s"][:,:,:,292:317].reshape(obs["priv_s"].size(0),batch_size,2,5,5)
            fireworks_empty_mask = (fireworks.sum(-1, keepdim=True) == 0.0)
            fireworks = torch.cat([fireworks, fireworks_empty_mask.float()], -1)
            fireworks = fireworks.argmax(dim = 4)

            for c in range(5):
                fireworks[:,:,:,c] = 67+5*c+fireworks[:,:,:,c]

            # 93-101 is info tokens

            info_tokens = 93 + torch.sum(obs["priv_s"][:,:,:,317:325], -1, dtype = torch.long)

            # 102-105 is life tokens

            life_tokens = 102 + torch.sum(obs["priv_s"][:,:,:,325:328], -1, dtype = torch.long)

            #

            if torch.sum(obs["priv_s"][1:,:,:,378:431]).item() == 0:
                move_type = torch.ones(obs["priv_s"].size(0), obs["priv_s"].size(1), obs["priv_s"].size(2), dtype = torch.long, device="cuda:1") * 203
                move_affect = torch.ones(obs["priv_s"].size(0), obs["priv_s"].size(1), obs["priv_s"].size(2), dtype = torch.long, device="cuda:1") * 204

            else:
                move_type = obs["priv_s"][1:,:,:,380:384]
                move_type_empty_mask = (move_type.sum(-1, keepdim=True) == 0.0)
                move_type = torch.cat([move_type, move_type_empty_mask.float()], -1)
                move_type = move_type.argmax(dim = 3)
                move_type = 5*move_type + 106

                which_colour = obs["priv_s"][1:,:,:,386:391]
                which_rank = obs["priv_s"][1:,:,:,391:396]
                which_play_disc = obs["priv_s"][1:,:,:,401:406]

                which_colour_empty_mask = (which_colour.sum(-1, keepdim=True) == 0.0)
                which_colour = torch.cat([which_colour_empty_mask.float(), which_colour], -1)

                which_rank_empty_mask = (which_rank.sum(-1, keepdim=True) == 0.0)
                which_rank = torch.cat([which_rank_empty_mask.float(), which_rank], -1)

                which_play_disc_empty_mask = (which_play_disc.sum(-1, keepdim=True) == 0.0)
                which_play_disc = torch.cat([which_play_disc_empty_mask.float(), which_play_disc], -1)

                which_colour = which_colour.argmax(dim = 3)
                which_rank = which_rank.argmax(dim = 3)
                which_play_disc = which_play_disc.argmax(dim = 3)

                move_type += (which_colour + which_rank + which_play_disc - 1)

                which_player = obs["priv_s"][1:,:,:,378:380]
                which_player_empty_mask = (which_player.sum(-1, keepdim=True) == 0.0)
                which_player = torch.cat([which_player, which_player_empty_mask.float()], -1)
                which_player = which_player.argmax(dim = 3)

                move_type += (20*which_player)

                move_affect = obs["priv_s"][1:,:,:,406:431]
                move_affect_empty_mask = (move_affect.sum(-1, keepdim=True) == 0.0)
                move_affect = torch.cat([move_affect, move_affect_empty_mask.float()], -1)
                move_affect = move_affect.argmax(dim = 3)

                move_affect += 146

                move_affect += (obs["priv_s"][1:,:,:,396:401].matmul(2**torch.arange(5, dtype=torch.float, device="cuda:1").flip(0).view(5,1))).reshape(-1, batch_size, 2).to(torch.long)

                move_type = torch.cat([torch.tensor([203 for _ in range(batch_size*2)], device="cuda:1", dtype=torch.long).reshape(1,batch_size,2), move_type], 0)
                move_affect = torch.cat([torch.tensor([204 for _ in range(batch_size*2)], device="cuda:1", dtype=torch.long).reshape(1,batch_size,2), move_affect.to(torch.long)], 0)

            stacked = torch.stack([partner_cards[:,:,:,0], partner_cards[:,:,:,1], partner_cards[:,:,:,2], 
                                    partner_cards[:,:,:,3], partner_cards[:,:,:,4], decksizes, fireworks[:,:,:,0],
                                    fireworks[:,:,:,1], fireworks[:,:,:,2], fireworks[:,:,:,3], fireworks[:,:,:,4], 
                                    info_tokens, life_tokens, move_type, move_affect] , dim=3)

            interleaved = torch.flatten(stacked, start_dim = 0, end_dim = 2)

            # for j in range(batch_size):
            #     for k in range(obs["priv_s"].size(0)):
            #         if torch.sum(obs["priv_s"][k, j, 0, :]) == 0:
            #             interleaved[j, :, k*15:] = 205
            #             break

            ### ouch
            # priv_s = obs["priv_s"].flatten(0, 1)
            priv_s = interleaved.reshape(-1, 15)
            legal_move = obs["legal_move"].flatten(0, 1)
            eps = obs["eps"].flatten(0, 1)

        hid = {
            "h0": obs["h0"].flatten(0, 1).transpose(0, 1).contiguous(),
            "c0": obs["c0"].flatten(0, 1).transpose(0, 1).contiguous(),
        }

        greedy_action, new_hid = self.greedy_act(priv_s.float(), legal_move, hid)

        random_action = legal_move.multinomial(1).squeeze(1)
        rand = torch.rand(greedy_action.size(), device=greedy_action.device)
        assert rand.size() == eps.size()
        rand = (rand < eps).long()
        action = (greedy_action * (1 - rand) + random_action * rand).detach().long()

        if self.vdn:
            action = action.view(obsize, ibsize, num_player)
            greedy_action = greedy_action.view(obsize, ibsize, num_player)
            rand = rand.view(obsize, ibsize, num_player)
        else:
            action = action.view(obsize, ibsize)
            greedy_action = greedy_action.view(obsize, ibsize)
            rand = rand.view(obsize, ibsize)

        hid_shape = (
            obsize,
            ibsize * num_player,
            self.online_net.num_lstm_layer,
            self.online_net.hid_dim,
        )
        h0 = new_hid["h0"].transpose(0, 1).view(*hid_shape)
        c0 = new_hid["c0"].transpose(0, 1).view(*hid_shape)

        reply = {
            "a": action.detach().cpu(),
            "greedy_a": greedy_action.detach().cpu(),
            "h0": h0.contiguous().detach().cpu(),
            "c0": c0.contiguous().detach().cpu(),
        }
        return reply

    @torch.jit.script_method
    def compute_priority(
        self, input_: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        compute priority for one batch
        """

        if self.uniform_priority:
            return {"priority": torch.ones_like(input_["reward"]).detach().cpu()}

        obsize, ibsize, num_player = 0, 0, 0
        flatten_end = 0
        if self.vdn:
            obsize, ibsize, num_player = input_["priv_s"].size()[:3]
            flatten_end = 2
        else:
            obsize, ibsize = input_["priv_s"].size()[:2]
            num_player = 1
            flatten_end = 1

        ### here we go
        batch_size = input_["priv_s"].size(1)

        obs = input_["priv_s"]

        # 0-25 is partner's cards
                #   0-25 is first card
                #   26-51 is second card
                #   52-77 is third card
                #   78-103 is fourth card
                #   104-129 is fifth card
        
        partner_cards = obs[:,:,:,125:250].reshape(obs.size(0),batch_size,2,5,25)
        partner_cards_empty_mask = (partner_cards.sum(-1, keepdim=True) == 0.0)
        partner_cards = torch.cat([partner_cards, partner_cards_empty_mask.float()], -1)
        partner_cards = partner_cards.argmax(dim = 4)

        # 26-66 is remaining deck size

        decksizes = 26 + torch.sum(obs[:,:,:,252:292], -1, dtype = torch.long)

        # 67+5*c-72+5*c is fireworks of colour c

        fireworks = obs[:,:,:,292:317].reshape(obs.size(0),batch_size,2,5,5)
        fireworks_empty_mask = (fireworks.sum(-1, keepdim=True) == 0.0)
        fireworks = torch.cat([fireworks, fireworks_empty_mask.float()], -1)
        fireworks = fireworks.argmax(dim = 4)

        for c in range(5):
            fireworks[:,:,:,c] = 67+5*c+fireworks[:,:,:,c]

        # 93-101 is info tokens

        info_tokens = 93 + torch.sum(obs[:,:,:,317:325], -1, dtype = torch.long)

        # 102-105 is life tokens

        life_tokens = 102 + torch.sum(obs[:,:,:,325:328], -1, dtype = torch.long)

        #

        if torch.sum(obs[1:,:,:,378:431]).item() == 0:
            move_type = torch.ones(obs.size(0), obs.size(1), obs.size(2), dtype = torch.long, device="cuda:1") * 203
            move_affect = torch.ones(obs.size(0), obs.size(1), obs.size(2), dtype = torch.long, device="cuda:1") * 204

        else:
            move_type = obs[1:,:,:,380:384]
            move_type_empty_mask = (move_type.sum(-1, keepdim=True) == 0.0)
            move_type = torch.cat([move_type, move_type_empty_mask.float()], -1)
            move_type = move_type.argmax(dim = 3)
            move_type = 5*move_type + 106

            which_colour = obs[1:,:,:,386:391]
            which_rank = obs[1:,:,:,391:396]
            which_play_disc = obs[1:,:,:,401:406]

            which_colour_empty_mask = (which_colour.sum(-1, keepdim=True) == 0.0)
            which_colour = torch.cat([which_colour_empty_mask.float(), which_colour], -1)

            which_rank_empty_mask = (which_rank.sum(-1, keepdim=True) == 0.0)
            which_rank = torch.cat([which_rank_empty_mask.float(), which_rank], -1)

            which_play_disc_empty_mask = (which_play_disc.sum(-1, keepdim=True) == 0.0)
            which_play_disc = torch.cat([which_play_disc_empty_mask.float(), which_play_disc], -1)

            which_colour = which_colour.argmax(dim = 3)
            which_rank = which_rank.argmax(dim = 3)
            which_play_disc = which_play_disc.argmax(dim = 3)

            move_type += (which_colour + which_rank + which_play_disc - 1)

            which_player = obs[1:,:,:,378:380]
            which_player_empty_mask = (which_player.sum(-1, keepdim=True) == 0.0)
            which_player = torch.cat([which_player, which_player_empty_mask.float()], -1)
            which_player = which_player.argmax(dim = 3)

            move_type += (20*which_player)

            move_affect = obs[1:,:,:,406:431]
            move_affect_empty_mask = (move_affect.sum(-1, keepdim=True) == 0.0)
            move_affect = torch.cat([move_affect, move_affect_empty_mask.float()], -1)
            move_affect = move_affect.argmax(dim = 3)

            move_affect += 146

            move_affect += (obs[1:,:,:,396:401].matmul(2**torch.arange(5, dtype=torch.float, device="cuda:1").flip(0).view(5,1))).reshape(-1, batch_size, 2).to(torch.long)

            move_type = torch.cat([torch.tensor([203 for _ in range(batch_size*2)], device="cuda:1", dtype=torch.long).reshape(1,batch_size,2), move_type], 0)
            move_affect = torch.cat([torch.tensor([204 for _ in range(batch_size*2)], device="cuda:1", dtype=torch.long).reshape(1,batch_size,2), move_affect.to(torch.long)], 0)

        stacked = torch.stack([partner_cards[:,:,:,0], partner_cards[:,:,:,1], partner_cards[:,:,:,2], 
                                partner_cards[:,:,:,3], partner_cards[:,:,:,4], decksizes, fireworks[:,:,:,0],
                                fireworks[:,:,:,1], fireworks[:,:,:,2], fireworks[:,:,:,3], fireworks[:,:,:,4], 
                                info_tokens, life_tokens, move_type, move_affect] , dim=3)

        interleaved = torch.flatten(stacked, start_dim = 0, end_dim = flatten_end)

        # for j in range(batch_size):
        #     for k in range(input_["priv_s"].size(0)):
        #         if torch.sum(input_["priv_s"][k, j, 0, :]) == 0:
        #             interleaved[j, :, k*15:] = 205
        #             break

        ### ouch

        priv_s = interleaved.reshape(-1, 15)
        # priv_s = input_["priv_s"].flatten(0, flatten_end)
        legal_move = input_["legal_move"].flatten(0, flatten_end)
        online_a = input_["a"].flatten(0, flatten_end)

        ### here we go
        batch_size = input_["next_priv_s"].size(1)

        obs = input_["next_priv_s"]

        # 0-25 is partner's cards
                #   0-25 is first card
                #   26-51 is second card
                #   52-77 is third card
                #   78-103 is fourth card
                #   104-129 is fifth card
        
        partner_cards = obs[:,:,:,125:250].reshape(obs.size(0),batch_size,2,5,25)
        partner_cards_empty_mask = (partner_cards.sum(-1, keepdim=True) == 0.0)
        partner_cards = torch.cat([partner_cards, partner_cards_empty_mask.float()], -1)
        partner_cards = partner_cards.argmax(dim = 4)

        # 26-66 is remaining deck size

        decksizes = 26 + torch.sum(obs[:,:,:,252:292], -1, dtype = torch.long)

        # 67+5*c-72+5*c is fireworks of colour c

        fireworks = obs[:,:,:,292:317].reshape(obs.size(0),batch_size,2,5,5)
        fireworks_empty_mask = (fireworks.sum(-1, keepdim=True) == 0.0)
        fireworks = torch.cat([fireworks, fireworks_empty_mask.float()], -1)
        fireworks = fireworks.argmax(dim = 4)

        for c in range(5):
            fireworks[:,:,:,c] = 67+5*c+fireworks[:,:,:,c]

        # 93-101 is info tokens

        info_tokens = 93 + torch.sum(obs[:,:,:,317:325], -1, dtype = torch.long)

        # 102-105 is life tokens

        life_tokens = 102 + torch.sum(obs[:,:,:,325:328], -1, dtype = torch.long)

        #

        if torch.sum(obs[1:,:,:,378:431]).item() == 0:
            move_type = torch.ones(obs.size(0), obs.size(1), obs.size(2), dtype = torch.long, device="cuda:1") * 203
            move_affect = torch.ones(obs.size(0), obs.size(1), obs.size(2), dtype = torch.long, device="cuda:1") * 204

        else:
            move_type = obs[1:,:,:,380:384]
            move_type_empty_mask = (move_type.sum(-1, keepdim=True) == 0.0)
            move_type = torch.cat([move_type, move_type_empty_mask.float()], -1)
            move_type = move_type.argmax(dim = 3)
            move_type = 5*move_type + 106

            which_colour = obs[1:,:,:,386:391]
            which_rank = obs[1:,:,:,391:396]
            which_play_disc = obs[1:,:,:,401:406]

            which_colour_empty_mask = (which_colour.sum(-1, keepdim=True) == 0.0)
            which_colour = torch.cat([which_colour_empty_mask.float(), which_colour], -1)

            which_rank_empty_mask = (which_rank.sum(-1, keepdim=True) == 0.0)
            which_rank = torch.cat([which_rank_empty_mask.float(), which_rank], -1)

            which_play_disc_empty_mask = (which_play_disc.sum(-1, keepdim=True) == 0.0)
            which_play_disc = torch.cat([which_play_disc_empty_mask.float(), which_play_disc], -1)

            which_colour = which_colour.argmax(dim = 3)
            which_rank = which_rank.argmax(dim = 3)
            which_play_disc = which_play_disc.argmax(dim = 3)

            move_type += (which_colour + which_rank + which_play_disc - 1)

            which_player = obs[1:,:,:,378:380]
            which_player_empty_mask = (which_player.sum(-1, keepdim=True) == 0.0)
            which_player = torch.cat([which_player, which_player_empty_mask.float()], -1)
            which_player = which_player.argmax(dim = 3)

            move_type += (20*which_player)

            move_affect = obs[1:,:,:,406:431]
            move_affect_empty_mask = (move_affect.sum(-1, keepdim=True) == 0.0)
            move_affect = torch.cat([move_affect, move_affect_empty_mask.float()], -1)
            move_affect = move_affect.argmax(dim = 3)

            move_affect += 146

            move_affect += (obs[1:,:,:,396:401].matmul(2**torch.arange(5, dtype=torch.float, device="cuda:1").flip(0).view(5,1))).reshape(-1, batch_size, 2).to(torch.long)

            move_type = torch.cat([torch.tensor([203 for _ in range(batch_size*2)], device="cuda:1", dtype=torch.long).reshape(1,batch_size,2), move_type], 0)
            move_affect = torch.cat([torch.tensor([204 for _ in range(batch_size*2)], device="cuda:1", dtype=torch.long).reshape(1,batch_size,2), move_affect.to(torch.long)], 0)

        stacked = torch.stack([partner_cards[:,:,:,0], partner_cards[:,:,:,1], partner_cards[:,:,:,2], 
                                partner_cards[:,:,:,3], partner_cards[:,:,:,4], decksizes, fireworks[:,:,:,0],
                                fireworks[:,:,:,1], fireworks[:,:,:,2], fireworks[:,:,:,3], fireworks[:,:,:,4], 
                                info_tokens, life_tokens, move_type, move_affect] , dim=3)

        interleaved = torch.flatten(stacked, start_dim = 0, end_dim = flatten_end)

        # for j in range(batch_size):
        #     for k in range(input_["next_priv_s"].size(0)):
        #         if torch.sum(input_["next_priv_s"][k, j, 0, :]) == 0:
        #             interleaved[j, :, k*15:] = 205
        #             break
        ### ouch

        next_priv_s = interleaved.reshape(-1, 15)
        # next_priv_s = input_["next_priv_s"].flatten(0, flatten_end)
        next_legal_move = input_["next_legal_move"].flatten(0, flatten_end)
        temperature = input_["temperature"].flatten(0, flatten_end)

        hid = {
            "h0": input_["h0"].flatten(0, 1).transpose(0, 1).contiguous(),
            "c0": input_["c0"].flatten(0, 1).transpose(0, 1).contiguous(),
        }   
        next_hid = {
            "h0": input_["next_h0"].flatten(0, 1).transpose(0, 1).contiguous(),
            "c0": input_["next_c0"].flatten(0, 1).transpose(0, 1).contiguous(),
        }
        reward = input_["reward"].flatten(0, 1)
        bootstrap = input_["bootstrap"].flatten(0, 1)

        online_qa = self.online_net(priv_s.float(), legal_move, online_a, hid)[0]
        next_a, _ = self.greedy_act(next_priv_s.float(), next_legal_move, next_hid)
        target_qa, _, _, _ = self.target_net(
            next_priv_s, next_legal_move, next_a, next_hid,
        )

        bsize = obsize * ibsize
        if self.vdn:
            # sum over action & player
            online_qa = online_qa.view(bsize, num_player).sum(1)
            target_qa = target_qa.view(bsize, num_player).sum(1)

        assert reward.size() == bootstrap.size()
        assert reward.size() == target_qa.size()
        target = reward + bootstrap * (self.gamma ** self.multi_step) * target_qa
        priority = (target - online_qa).abs()
        priority = priority.view(obsize, ibsize).detach().cpu()
        return {"priority": priority}

    ############# python only functions #############
    def flat_4d(self, data):
        """
        rnn_hid: [num_layer, batch, num_player, dim] -> [num_player, batch, dim]
        seq_obs: [seq_len, batch, num_player, dim] -> [seq_len, batch, dim]
        """
        bsize = 0
        num_player = 0
        for k, v in data.items():
            if num_player == 0:
                bsize, num_player = v.size()[1:3]

            if v.dim() == 4:
                d0, d1, d2, d3 = v.size()
                data[k] = v.view(d0, d1 * d2, d3)
            elif v.dim() == 3:
                d0, d1, d2 = v.size()
                data[k] = v.view(d0, d1 * d2)
        return bsize, num_player

    def td_error(self, obs, hid, action, reward, terminal, bootstrap, seq_len, stat):
        max_seq_len = obs["priv_s"].size(0)

        bsize, num_player = 0, 1
        if self.vdn:
            bsize, num_player = self.flat_4d(obs)
            self.flat_4d(action)

        # priv_s = obs["priv_s"]
        ### here we go
        batch_size = obs["priv_s"].size(1)

        # 0-25 is partner's cards
                #   0-25 is first card
                #   26-51 is second card
                #   52-77 is third card
                #   78-103 is fourth card
                #   104-129 is fifth card
        
        partner_cards = obs["priv_s"][:,:,125:250].reshape(obs["priv_s"].size(0),batch_size,5,25)
        partner_cards_empty_mask = (partner_cards.sum(-1, keepdim=True) == 0.0)
        partner_cards = torch.cat([partner_cards, partner_cards_empty_mask.float()], -1)
        partner_cards = partner_cards.argmax(dim = 3)

        # 26-66 is remaining deck size

        decksizes = 26 + torch.sum(obs["priv_s"][:,:,252:292], -1, dtype = torch.long)

        # 67+5*c-72+5*c is fireworks of colour c

        fireworks = obs["priv_s"][:,:,292:317].reshape(obs["priv_s"].size(0),batch_size,5,5)
        fireworks_empty_mask = (fireworks.sum(-1, keepdim=True) == 0.0)
        fireworks = torch.cat([fireworks, fireworks_empty_mask.float()], -1)
        fireworks = fireworks.argmax(dim = 3)

        for c in range(5):
            fireworks[:,:,c] = 67+5*c+fireworks[:,:,c]

        # 93-101 is info tokens

        info_tokens = 93 + torch.sum(obs["priv_s"][:,:,317:325], -1, dtype = torch.long)

        # 102-105 is life tokens

        life_tokens = 102 + torch.sum(obs["priv_s"][:,:,325:328], -1, dtype = torch.long)

        #

        if torch.sum(obs["priv_s"][1:,:,378:431]).item() == 0:
            move_type = torch.ones(obs["priv_s"].size(0), obs["priv_s"].size(1), obs["priv_s"].size(2), dtype = torch.long, device="cuda:1") * 203
            move_affect = torch.ones(obs["priv_s"].size(0), obs["priv_s"].size(1), obs["priv_s"].size(2), dtype = torch.long, device="cuda:1") * 204

        else:
            move_type = obs["priv_s"][1:,:,380:384]
            move_type_empty_mask = (move_type.sum(-1, keepdim=True) == 0.0)
            move_type = torch.cat([move_type, move_type_empty_mask.float()], -1)
            move_type = move_type.argmax(dim = 2)
            move_type = 5*move_type + 106

            which_colour = obs["priv_s"][1:,:,386:391]
            which_rank = obs["priv_s"][1:,:,391:396]
            which_play_disc = obs["priv_s"][1:,:,401:406]

            which_colour_empty_mask = (which_colour.sum(-1, keepdim=True) == 0.0)
            which_colour = torch.cat([which_colour_empty_mask.float(), which_colour], -1)

            which_rank_empty_mask = (which_rank.sum(-1, keepdim=True) == 0.0)
            which_rank = torch.cat([which_rank_empty_mask.float(), which_rank], -1)

            which_play_disc_empty_mask = (which_play_disc.sum(-1, keepdim=True) == 0.0)
            which_play_disc = torch.cat([which_play_disc_empty_mask.float(), which_play_disc], -1)

            which_colour = which_colour.argmax(dim = 2)
            which_rank = which_rank.argmax(dim = 2)
            which_play_disc = which_play_disc.argmax(dim = 2)

            move_type += (which_colour + which_rank + which_play_disc - 1)

            which_player = obs["priv_s"][1:,:,378:380]
            which_player_empty_mask = (which_player.sum(-1, keepdim=True) == 0.0)
            which_player = torch.cat([which_player, which_player_empty_mask.float()], -1)
            which_player = which_player.argmax(dim = 2)

            move_type += (20*which_player)

            move_affect = obs["priv_s"][1:,:,406:431]
            move_affect_empty_mask = (move_affect.sum(-1, keepdim=True) == 0.0)
            move_affect = torch.cat([move_affect, move_affect_empty_mask.float()], -1)
            move_affect = move_affect.argmax(dim = 2)

            move_affect += 146

            move_affect += (obs["priv_s"][1:,:,396:401].matmul(2**torch.arange(5, dtype=torch.float, device="cuda:0").flip(0).view(5,1))).reshape(-1, batch_size).to(torch.long)

            move_type = torch.cat([torch.tensor([203 for _ in range(batch_size)], device="cuda:0", dtype=torch.long).reshape(1,batch_size), move_type], 0)
            move_affect = torch.cat([torch.tensor([204 for _ in range(batch_size)], device="cuda:0", dtype=torch.long).reshape(1,batch_size), move_affect.to(torch.long)], 0)

        stacked = torch.stack([partner_cards[:,:,0], partner_cards[:,:,1], partner_cards[:,:,2], 
                                partner_cards[:,:,3], partner_cards[:,:,4], decksizes, fireworks[:,:,0],
                                fireworks[:,:,1], fireworks[:,:,2], fireworks[:,:,3], fireworks[:,:,4], 
                                info_tokens, life_tokens, move_type, move_affect] , dim=2)


        # interleaved = torch.flatten(stacked, start_dim = 0, end_dim = 1)

        for j in range(batch_size):
            stacked[int(seq_len[j % (batch_size // 2)]):, j, :] = 205

        ### ouch
        priv_s = stacked
        legal_move = obs["legal_move"]
        action = action["a"]

        hid = {}

        # this only works because the trajectories are padded,
        # i.e. no terminal in the middle
        online_qa, greedy_a, _, lstm_o = self.online_net(
            priv_s.float(), legal_move, action, hid
        )

        with torch.no_grad():
            target_qa, _, _, _ = self.target_net(priv_s, legal_move, greedy_a, hid)
            # assert target_q.size() == pa.size()
            # target_qe = (pa * target_q).sum(-1).detach()
            assert online_qa.size() == target_qa.size()

        if self.vdn:
            online_qa = online_qa.view(max_seq_len, bsize, num_player).sum(-1)
            target_qa = target_qa.view(max_seq_len, bsize, num_player).sum(-1)
            lstm_o = lstm_o.view(max_seq_len, bsize, num_player, -1)

        terminal = terminal.float()
        bootstrap = bootstrap.float()

        errs = []
        target_qa = torch.cat(
            [target_qa[self.multi_step :], target_qa[: self.multi_step]], 0
        )
        target_qa[-self.multi_step :] = 0

        assert target_qa.size() == reward.size()
        target = reward + bootstrap * (self.gamma ** self.multi_step) * target_qa
        mask = torch.arange(0, max_seq_len, device=seq_len.device)
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
        err = (target.detach() - online_qa) * mask
        return err, lstm_o

    def aux_task_iql(self, lstm_o, hand, seq_len, rl_loss_size, stat):
        seq_size, bsize, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, self.online_net.hand_size, 3)
        own_hand_slot_mask = own_hand.sum(3)
        pred_loss1, avg_xent1, _, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size

        stat["aux1"].feed(avg_xent1)
        return pred_loss1

    def aux_task_vdn(self, lstm_o, hand, t, seq_len, rl_loss_size, stat):
        """1st and 2nd order aux task used in VDN"""
        seq_size, bsize, num_player, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, num_player, self.online_net.hand_size, 3)
        own_hand_slot_mask = own_hand.sum(4)
        pred_loss1, avg_xent1, belief1, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size

        rotate = [num_player - 1]
        rotate.extend(list(range(num_player - 1)))
        partner_hand = own_hand[:, :, rotate, :, :]
        partner_hand_slot_mask = partner_hand.sum(4)
        partner_belief1 = belief1[:, :, rotate, :, :].detach()

        stat["aux1"].feed(avg_xent1)
        return pred_loss1

    def loss(self, batch, pred_weight, stat):
        err, lstm_o = self.td_error(
            batch.obs,
            batch.h0,
            batch.action,
            batch.reward,
            batch.terminal,
            batch.bootstrap,
            batch.seq_len,
            stat,
        )
        rl_loss = nn.functional.smooth_l1_loss(
            err, torch.zeros_like(err), reduction="none"
        )
        rl_loss = rl_loss.sum(0)
        stat["rl_loss"].feed((rl_loss / batch.seq_len).mean().item())

        priority = err.abs()
        # priority = self.aggregate_priority(p, batch.seq_len)

        if pred_weight > 0:
            if self.vdn:
                pred_loss1 = self.aux_task_vdn(
                    lstm_o,
                    batch.obs["own_hand"],
                    batch.obs["temperature"],
                    batch.seq_len,
                    rl_loss.size(),
                    stat,
                )
                loss = rl_loss + pred_weight * pred_loss1
            else:
                pred_loss = self.aux_task_iql(
                    lstm_o, batch.obs["own_hand"], batch.seq_len, rl_loss.size(), stat,
                )
                loss = rl_loss + pred_weight * pred_loss
        else:
            loss = rl_loss
        return loss, priority
