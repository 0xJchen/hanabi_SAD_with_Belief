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
import random
import copy
class BeliefEncoder(nn.Module):
#class BeliefEncoder(torch.jit.ScriptModule):
    def __init__(self, input_dim, num_embed_layer, hid_dim, n_layers=1, dropout=0.0):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.num_embed_layer = num_embed_layer
        self.input_dim = input_dim
        self.rnn = nn.LSTM(hid_dim, hid_dim, n_layers, dropout=dropout)
        self.num_ff_layer=1
        self.dropout = nn.Dropout(dropout)

        ff_layers = [nn.Linear(self.input_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.embed_net = nn.Sequential(*ff_layers)

    def forward(self, src,encoder_hid):
        #src [batch, input_dim]
        embedded = self.dropout(self.embed_net(src))
        if len(encoder_hid) ==0:
            outputs, (hidden, cell) = self.rnn(embedded)
        else:
            output,(hidden,cell)=self.rnn(embedded,(encoder_hid["priv_hidden"],encoder_hid["priv_cell"]))

        return hidden, cell

class BeliefDecoder(nn.Module):
#class BeliefDecoder(torch.jit.ScriptModule):
    def __init__(self, hid_dim, output_dim, device,card_slot_dim=25, n_layers=1, dropout=0.0):  # output_dim is total cards dim
        super().__init__()
        self.device=device
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(self.output_dim, self.hid_dim).to(self.device)
        self.rnn = nn.LSTM(hid_dim*2+card_slot_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, card_slot_dim)
        self.dropout = nn.Dropout(dropout).to(self.device)
        # self.softmax=nn.Softmax(dim=1) unnecessary, as crossentropy remedies this

    def forward(self, predicted_hand, cxt_vec, decoder_hid):
        #predicted_hand [batch,25]->[batch,hid=512]
        #print("before embedding",predicted_hand.shape)
        #predicted_hand = self.dropout(self.embedding(predicted_hand.long()))
        #print("in decoder forward",predicted_hand.shape,cxt_vec.shape)
        if predicted_hand.dim() == 2:
            predicted_hand=predicted_hand.unsqueeze(0)
        unified_input = torch.cat((predicted_hand, cxt_vec),dim=2)
        assert unified_input.dim()==3
        #unified_input [1,batch,hid=512*3]
        if len(decoder_hid) == 0:
            output, (hidden, cell) = self.rnn(unified_input)
        else:
            output, (hidden, cell) = self.rnn(unified_input, (decoder_hid["decoder_hidden"],decoder_hid["decoder_cell"]))
        prediction = self.fc_out(output)
        # predicted = self.softmax(prediction)
        #prediction [batch,25]
        return prediction, hidden, cell

class BeliefModel(nn.Module):
#class BeliefModel(torch.jit.ScriptModule):
    def __init__(self, encoder, decoder, device, batch_size,hand_size=5, card_size=25, teacher_forcing_ratio=1):
        super().__init__()
        self.device = device
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.hand_size = hand_size
        self.batch_size = batch_size
        print("gg",batch_size)
        self.card_size = card_size
        self.teacher_forcing_ratio=teacher_forcing_ratio
        self.loss_criterion = nn.CrossEntropyLoss(reduction='none')

    def flat_4d(self, data):
        """
        rnn_hid: [num_layer, batch, num_player, dim] -> [num_player, batch, dim]
        seq_obs: [seq_len, batch, num_player, dim] -> [seq_len, batch, dim]
        """
#        bsize = 0
#        num_player = 0
#        for k, v in data.items():
            # if num_player == 0:
            #     bsize, num_player = v.size()[1:3]

#            if v.dim() == 4:
#                d0, d1, d2, d3 = v.size()
#                data[k] = v.view(d0, d1 * d2, d3)
#            elif v.dim() == 3:
#               d0, d1, d2 = v.size()
#                data[k] = v.view(d0, d1 * d2)
#        return bsize, num_player
        d0,d1,d2,d3=data.size()
        return data.view(d0,d1*d2,d3)

    def forward(self, src, trg, encoder_hid):
        # trg = [batch_size,hand_Size, card_size]
        # trg = [1, batch,card_slot*card_size]
        assert trg.dim()==3#1,batch,125
        cur_trg=copy.deepcopy(trg)
        cur_trg=cur_trg.split(25,dim=-1)
        #print("in forward: ",src.shape,trg.shape)
        #seq_length, batch_size,_=trg.shape
        #trg=trg.view(seq_length,batch_size,self.hand_size,self.card_size)
        hidden,cell=self.encoder(src,encoder_hid)
        assert hidden.dim()==3
        cxt_vec = torch.cat((hidden, cell),2)

        prediction = torch.zeros((self.batch_size, self.hand_size, self.card_size)).to(self.device)

        predicted_hand = torch.zeros((self.batch_size, self.card_size)).to(self.device)
        prediction_input = torch.zeros((self.batch_size, self.card_size)).to(self.device)
        #print("batch size: ",self.batch_size)
        decoder_hid={}
        for hand_idx in range(self.hand_size):#must teacher-forcing
            #print("in loss: calculating hand idx: ",hand_idx)
            if hand_idx == 0:
                predicted_hand, decoder_hid["decoder_hidden"], decoder_hid["decoder_cell"] = self.decoder(
                    prediction_input, cxt_vec,decoder_hid)
            else:
                predicted_hand,decoder_hid["decoder_hidden"], decoder_hid["decoder_cell"] = self.decoder(
                    prediction_input, cxt_vec, decoder_hid)

            #save predicted hand slot
            prediction[:,hand_idx,:] = predicted_hand
            prediction_input=cur_trg[hand_idx]
            # teacher_force=random.random() < self.teacher_forcing_ratio
            # prediction_input=trg[:,hand_idx,:] if teacher_forcing else predicted_hand
        prediction=prediction.view(self.batch_size,-1)
       # print("prediction baby: ",type(prediction))
        return prediction, (hidden,cell)
    def loss(self, batch,stat):
        ground_truth=batch.obs["sl_hand"]
        #sl_hand shape:  torch.Size([80, 64, 2, 125])
        # print("sl_hand shape: ",ground_truth.shape)
        obs=batch.obs["priv_s"]
        # print("obs shape",obs.shape)
        obs=self.flat_4d(obs)
        ground_truth=self.flat_4d(ground_truth)
        #shape: [seq,batch,H]
        loss=torch.zeros((80,128))
        hid={}

        for seq_idx in range(obs.shape[0]):
            prediction,(hid["priv_hidden"],hid["priv_cell"])=self.forward(obs[seq_idx].unsqueeze(0),ground_truth[seq_idx].unsqueeze(0),hid)
            cur_ground_truth=ground_truth[seq_idx].reshape(-1,5,25).permute(0,2,1)#[128,25,5]
            cur_ground_truth=cur_ground_truth.argmax(dim=1)#[128,5]
            cur_loss=self.loss_criterion(prediction.view(-1,5,25).permute(0,2,1), cur_ground_truth)
            loss[seq_idx]=cur_loss.sum(dim=1)
            #if seq_idx == 0:
            #    print(cur_loss.shape,cur_loss.view(-1).shape)
            #else:
            #    prediction,(hid["priv_hidden"],hid["priv_cell"])=self.forward(obs[seq_idx].unsqueeze(0),ground_truth[seq_idx].unsqueeze(0),hid)
            #    cur_ground_truth=ground_truth[seq_idx].reshape(-1,5,25).permute(0,2,1)
            #    cur_ground_truth=cur_ground_truth.argmax(dim=1)
            #    loss[seq_idx]=self.loss_criterion(prediction.view(-1,5,25).permute(0,2,1), cur_ground_truth)
           # print("=====finish seq: =====",seq_idx)

        return loss,torch.ones_like(loss)

    def beliefQuality(self,batch):
        ground_truth=batch.obs["sl_hand"]
        #sl_hand shape:  torch.Size([80, 64, 2, 125])
        # print("sl_hand shape: ",ground_truth.shape)
        obs=batch.obs["priv_s"]
        # print("obs shape",obs.shape)
        obs=self.flat_4d(obs)
        ground_truth=self.flat_4d(ground_truth)
        #shape: [seq,batch,H]
        loss=torch.zeros(80)
        hid={}

        for seq_idx in range(obs.shape[0]):
            prediction,(hid["priv_hidden"],hid["priv_cell"])=self.forward(obs[seq_idx].unsqueeze(0),ground_truth[seq_idx].unsqueeze(0),hid)
            # print("prediction: ",prediction[1])
            cur_ground_truth=ground_truth[seq_idx].reshape(-1,5,25).permute(0,2,1)#[128,25,5]
            cur_ground_truth=cur_ground_truth.argmax(dim=1)#[128,5]
            cur_loss=self.loss_criterion(prediction.view(-1,5,25).permute(0,2,1), cur_ground_truth)
            loss[seq_idx]=cur_loss.mean()
            # print('seq id: {0}, loss: {1}'.format(seq_idx,cur_loss.mean()))
        return loss