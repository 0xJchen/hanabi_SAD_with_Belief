# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
from typing import Dict, Tuple
import pprint
import argparse

import torch
import torch.nn as nn

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(lib_path)
sys.path.append(lib_path)
import utils

class BeliefEncoder(torch.jit.ScriptModule):
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

    @torch.jit.script_method
    def forward(self, src:torch.Tensor ,encoder_hid:Dict[str, torch.Tensor] ) -> Tuple[torch.Tensor, torch.Tensor]:
        #src [batch, input_dim]
        embedded = self.dropout(self.embed_net(src))
        # print("embed: ",embedded.shape)
        # print("hid: ",encoder_hid["priv_hidden"].shape)
        if len(encoder_hid) ==0:
            outputs, (hidden, cell) = self.rnn(embedded)
        else:
            output,(hidden,cell)=self.rnn(embedded,(encoder_hid["priv_hidden"],encoder_hid["priv_cell"]))

        return hidden, cell

class BeliefDecoder(torch.jit.ScriptModule):
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
        self.softmax=nn.Softmax(dim=-1)#assume[batch,25]

    @torch.jit.script_method
    def forward(self, predicted_hand: torch.Tensor, cxt_vec: torch.Tensor, decoder_hid: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
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
        prediction=self.softmax(prediction)
        #prediction [batch,25]
        return prediction, hidden, cell

class BeliefModel(torch.jit.ScriptModule):
#class BeliefModel(torch.jit.ScriptModule):
    def __init__(self, encoder, decoder, device, batch_size,hand_size=5, card_size=25, teacher_forcing_ratio=1):
        super().__init__()
        self.device = device
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.hand_size = hand_size
        self.batch_size = batch_size
        self.card_size = card_size
        self.teacher_forcing_ratio=teacher_forcing_ratio
        self.loss_criterion = nn.CrossEntropyLoss(reduction='none')

    @torch.jit.script_method
    def flat_4d(self, data: torch.Tensor) -> torch.Tensor:
        """
        rnn_hid: [num_layer, batch, num_player, dim] -> [num_player, batch, dim]
        seq_obs: [seq_len, batch, num_player, dim] -> [seq_len, batch, dim]
        """
        # print("flat 4d: ",data.shape)
        if data.dim()==4:
            d0,d1,d2,d3=data.size()
            return data.view(d0,d1*d2,d3)
        elif data.dim()==3:
            d1,d2,d3=data.size()
            d0=1
            return data.view(d0,d1*d2,d3)
        elif data.dim()==2:
            # assert(data.dim()==2)
            d0=1
            d1=1
            d2,d3=data.size()
            return data.view(d0,d1*d2,d3)
        else:
            assert data.dim()==1
            d0=1
            d1=1
            d2=1
            d3=data.size()[0]
            return data.reshape(d0,d1*d2,d3)#[seq=1,batch*2,125]

    @torch.jit.script_method
    def predict_hand(self, src: torch.Tensor,  encoder_hid: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor,Tuple[torch.Tensor,torch.Tensor]]:
        # trg = [batch_size,hand_Size, card_size]
        # trg = [1, batch,card_slot*card_size]
        # assert trg.dim()==3#1,batch,125
        # # cur_trg=copy.deepcopy(trg)
        # cur_trg=trg.split(25,dim=-1)
        # #print("in forward: ",src.shape,trg.shape)
        # #seq_length, batch_size,_=trg.shape
        #trg=trg.view(seq_length,batch_size,self.hand_size,self.card_size)
        hidden,cell=self.encoder(src,encoder_hid)
        # print("finish encoder!")
        # assert hidden.dim()==3
        cxt_vec = torch.cat((hidden, cell),-1)

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
            # print("predict hand shape: ",predicted_hand.shape)
            #save predicted hand slot
            prediction[:,hand_idx,:] = predicted_hand.squeeze(1);
            # prediction_input=cur_trg[hand_idx]; 
            # teacher_force=random.random() < self.teacher_forcing_ratio
            # prediction_input=trg[:,hand_idx,:] if teacher_forcing else predicted_hand

        # prediction=prediction.view(self.batch_size,-1)

        return prediction, (hidden,cell)
        
    @torch.jit.script_method
    def forward(self, obs_with_hiden: Dict[str, torch.Tensor] ) -> Dict[str, torch.Tensor] :
        # ground_truth=batch.obs["sl_hand"]
        #sl_hand shape:  torch.Size([80, 64, 2, 125])
        # print("sl_hand shape: ",ground_truth.shape)
        obs=obs_with_hiden["s"].contiguous()
        # print("obs shape",obs.shape)
        obs=self.flat_4d(obs)
        self.batch_size=obs.shape[1]
        # who=obs_with_hiden["who"]
        # ground_truth=self.flat_4d(ground_truth)
        #shape: [seq,batch,H]
        # loss=torch.zeros((80,128))
        # hid={}
        output={}
        hid={}
        hid["priv_hidden"]=obs_with_hiden["h0"].contiguous()
        hid["priv_cell"]=obs_with_hiden["c0"].contiguous()
        
        #obs=> [1,batch,125]
        player_hand,(h,c)=self.predict_hand(obs,hid)
        output["hidden"]= h.contiguous()
        output["cell"]=c.contiguous()
        batchsize,handsize,cardsize=player_hand.shape
        assert(handsize==5 and cardsize==25)
        
        if batchsize != 1:
            player_hand=player_hand.view(batchsize//2,2,handsize,cardsize)
            output["hand_zero"]=player_hand[self.batch_size,0,:,:]
            output["hand_one"]=player_hand[self.batch_size,1,:,:]
        else:
            player_hand=player_hand.view(1,handsize,cardsize)
            output["hand"]=player_hand
        # player_hand=player_hand[self.batch_size,who,:,:]


        # output["hand"]=player_hand.view(self.batch_size,self.hand_size,self.card_size)

        return output
        # output["hand"]=


        # for seq_idx in range(obs.shape[0]):
        #     prediction,(hid["priv_hidden"],hid["priv_cell"])=self.forward(obs[seq_idx].unsqueeze(0),ground_truth[seq_idx].unsqueeze(0),hid)
        #     cur_ground_truth=ground_truth[seq_idx].reshape(-1,5,25).permute(0,2,1)#[128,25,5]
        #     cur_ground_truth=cur_ground_truth.argmax(dim=1)#[128,5]
        #     cur_loss=self.loss_criterion(prediction.view(-1,5,25).permute(0,2,1), cur_ground_truth)
        #     loss[seq_idx]=cur_loss.sum(dim=1)
        #     if seq_idx == 0:
        #        print(cur_loss.shape,cur_loss.view(-1).shape)
        #     else:
        #        prediction,(hid["priv_hidden"],hid["priv_cell"])=self.forward(obs[seq_idx].unsqueeze(0),ground_truth[seq_idx].unsqueeze(0),hid)
        #        cur_ground_truth=ground_truth[seq_idx].reshape(-1,5,25).permute(0,2,1)
        #        cur_ground_truth=cur_ground_truth.argmax(dim=1)
        #        loss[seq_idx]=self.loss_criterion(prediction.view(-1,5,25).permute(0,2,1), cur_ground_truth)

           # 
           # 
       
    # def loss(self, batch,stat):
    #     ground_truth=batch.obs["sl_hand"]
    #     #sl_hand shape:  torch.Size([80, 64, 2, 125])
    #     # print("sl_hand shape: ",ground_truth.shape)
    #     obs=batch.obs["priv_s"]
    #     print("obs shape",obs.shape)
    #     obs=self.flat_4d(obs)
    #     ground_truth=self.flat_4d(ground_truth)
    #     #shape: [seq,batch,H]
    #     loss=torch.zeros((80,128))
    #     hid={}

    #     for seq_idx in range(obs.shape[0]):
    #         prediction,(hid["priv_hidden"],hid["priv_cell"])=self.forward(obs[seq_idx].unsqueeze(0),ground_truth[seq_idx].unsqueeze(0),hid)
    #         cur_ground_truth=ground_truth[seq_idx].reshape(-1,5,25).permute(0,2,1)#[128,25,5]
    #         cur_ground_truth=cur_ground_truth.argmax(dim=1)#[128,5]
    #         cur_loss=self.loss_criterion(prediction.view(-1,5,25).permute(0,2,1), cur_ground_truth)
    #         loss[seq_idx]=cur_loss.sum(dim=1)
            #if seq_idx == 0:
            #    print(cur_loss.shape,cur_loss.view(-1).shape)
            #else:
            #    prediction,(hid["priv_hidden"],hid["priv_cell"])=self.forward(obs[seq_idx].unsqueeze(0),ground_truth[seq_idx].unsqueeze(0),hid)
            #    cur_ground_truth=ground_truth[seq_idx].reshape(-1,5,25).permute(0,2,1)
            #    cur_ground_truth=cur_ground_truth.argmax(dim=1)
            #    loss[seq_idx]=self.loss_criterion(prediction.view(-1,5,25).permute(0,2,1), cur_ground_truth)
           # print("=====finish seq: =====",seq_idx)

        # return loss,torch.ones_like(loss)

    
class LSTMNet(torch.jit.ScriptModule):
    def __init__(
        self,
        device,
        in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        hand_size,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_lstm_layer = num_lstm_layer

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            self.hid_dim, self.hid_dim, num_layers=self.num_lstm_layer
        ).to(device)
        self.lstm.flatten_parameters()
        
        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

    @torch.jit.script_method
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h0 = obs["h0"].transpose(0, 1).contiguous()
        c0 = obs["c0"].transpose(0, 1).contiguous()

        s = obs["s"].unsqueeze(0)
        assert s.size(2) == self.in_dim

        x = self.net(s)
        o, (h, c) = self.lstm(x, (h0, c0))
        a = self.fc_a(o).squeeze(0)

        return {
            "a": a,
            "h0": h.transpose(0, 1).contiguous(),
            "c0": c.transpose(0, 1).contiguous(),
        }


## main program ##
parser = argparse.ArgumentParser(description="")
parser.add_argument("--model", type=str, default=None)
args = parser.parse_args()


device = "cuda"
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder = os.path.join(root, "models", "op", "sad-aux-op")
lstmweight_file=os.path.join(folder, "M0.pthw")
folder = os.path.join(root, "pyhanabi", "exps", "lbf")
beliefWeight_file=os.path.join(folder, "model_epoch300.pthw")


if not os.path.exists(lstmweight_file):
    print(f"Cannot find weight at: {lstmweight_file}")
    assert False

state_dict = torch.load(lstmweight_file)
in_dim = state_dict["net.0.weight"].size()[1]
out_dim = state_dict["fc_a.weight"].size()[0]
print("out dim: ",out_dim)

print("after loading model")

# search_model = LSTMNet(device, in_dim, 512, out_dim, 2, 5)
# utils.load_weight(search_model, lstmweight_file, device)

Belief_Encoder=BeliefEncoder(838,1,256)
Belief_Decoder=BeliefDecoder(256,25,device)
belief_model=BeliefModel(Belief_Encoder,Belief_Decoder,device,400)
utils.load_weight(belief_model, beliefWeight_file, device)


# save_path="sparta.pth"
# print("saving search model to:", save_path)
# torch.jit.save(search_model, save_path)

save_path="belief.pth"
print("saving belief model to:", save_path)
torch.jit.save(belief_model, save_path)

mock_input={}
mock_input["s"]=torch.zeros(1,838).to(device)
mock_input["h0"]=torch.zeros((1,1,256)).to(device)
mock_input["c0"]=torch.zeros((1,1,256)).to(device)

out=belief_model.forward(mock_input)
print("output hid shape: ",out["hidden"].shape)