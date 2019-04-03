#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid.layers as layers
import paddle.fluid as fluid
import numpy as np
import ipdb

test_mode=False
random_seed=123
para_init=False
cell_clip=3.0
proj_clip=3.0
init1=0.1
hidden_size=4096
para_init=False
vocab_size=52445
emb_size=512
modify = 1
DEBUG = 0

def dropout(input):
    dropout1=0.1
    if modify == 1:
       dropout1=0.2
    return layers.dropout(
            input,
            dropout_prob=dropout1,
            dropout_implementation="upscale_in_train",
            is_test=False)


def lstmp_encoder(input_seq, gate_size, h_0, c_0, para_name, proj_size, test_mode, args):
    # A lstm encoder implementation with projection.
    # Linear transformation part for input gate, output gate, forget gate
    # and cell activation vectors need be done outside of dynamic_lstm.
    # So the output size is 4 times of gate_size.

    if para_init:
        init = fluid.initializer.Constant(init1)
        init_b = fluid.initializer.Constant(0.0)
    else:
        init = None
        init_b = None
    if modify==0:
        input_seq = dropout(input_seq)
    input_proj = layers.fc(input=input_seq,
                           param_attr=fluid.ParamAttr(
                               name=para_name + '_gate_w', initializer=init),
                           size=gate_size * 4,
                           act=None,
                           bias_attr=False)
    #layers.Print(input_seq, message='input_seq', summarize=10)
    #layers.Print(input_proj, message='input_proj', summarize=10)
    hidden, cell = layers.dynamic_lstmp(
        input=input_proj,
        size=gate_size * 4,
        proj_size=proj_size,
        h_0=h_0,
        c_0=c_0,
        use_peepholes=False,
        proj_clip=proj_clip,
        cell_clip=cell_clip,
        proj_activation="identity",
        param_attr=fluid.ParamAttr(initializer=init),
        bias_attr=fluid.ParamAttr(initializer=init_b))

    return hidden, cell, input_proj
def emb(x, vocab_size=52445,emb_size=512):
    x_emb = layers.embedding(
        input=x,
        size=[vocab_size, emb_size],
        dtype='float32',
        is_sparse=False,
        param_attr=fluid.ParamAttr(name='embedding_para'))
    return x_emb


def encoder_1(x_emb,
            vocab_size,
            emb_size,
            init_hidden=None,
            init_cell=None,
            para_name='',
            args=None):
    rnn_input = x_emb
    #rnn_input.stop_gradient = True
    rnn_outs = []
    rnn_outs_ori = []
    cells = []
    projs = []
    num_layers=2
    for i in range(num_layers):
        if modify==0:
            rnn_input = dropout(rnn_input)
        if init_hidden and init_cell:
            h0 = layers.squeeze(
                layers.slice(
                    init_hidden, axes=[0], starts=[i], ends=[i + 1]),
                axes=[0])
            c0 = layers.squeeze(
                layers.slice(
                    init_cell, axes=[0], starts=[i], ends=[i + 1]),
                axes=[0])
        else:
            h0 = c0 = None
        rnn_out, cell, input_proj = lstmp_encoder(
            rnn_input, hidden_size, h0, c0,
            para_name + 'layer{}'.format(i + 1), emb_size, test_mode, args)
        rnn_out_ori = rnn_out
        if i > 0:
            rnn_out = rnn_out + rnn_input
        if modify==0:
            rnn_out = dropout(rnn_out)
        rnn_out.stop_gradient = True
        rnn_outs.append(rnn_out)
        #rnn_outs_ori.stop_gradient = True
        rnn_outs_ori.append(rnn_out_ori)
    #ipdb.set_trace()
     #layers.Print(input_seq, message='input_seq', summarize=10)
    #layers.Print(rnn_outs[-1], message='rnn_outs', summarize=10)
    return  rnn_outs, rnn_outs_ori


def weight_layers(lm_embeddings, name="", l2_coef=0.0):
    '''
    Weight the layers of a biLM with trainable scalar weights to
    compute ELMo representations.

    Input:
        lm_embeddings(list): representations of 2 layers from biLM.
        name = a string prefix used for the trainable variable names
        l2_coef: the l2 regularization coefficient $\lambda$.
            Pass None or 0.0 for no regularization.

    Output:
        weighted_lm_layers: weighted embeddings form biLM
    '''

    n_lm_layers = len(lm_embeddings)
    W = layers.create_parameter([n_lm_layers, ], dtype="float32", name=name+"ELMo_w",
                                attr=fluid.ParamAttr(name=name+"ELMo_w",
                                                     initializer=fluid.initializer.Constant(0.0),
                                                     regularizer=fluid.regularizer.L2Decay(l2_coef)))
    if DEBUG:
        fluid.layers.Print(lm_embeddings[0], first_n=1, summarize=10, message="lm_embeddings_0")
        fluid.layers.Print(lm_embeddings[1], first_n=1, summarize=10, message="lm_embeddings_1")
        fluid.layers.Print(lm_embeddings[2], first_n=1, summarize=10, message="lm_embeddings_2")
        fluid.layers.Print(W, first_n=1, summarize=10, message="weights_before_sotfmax")
    normed_weights = layers.softmax( W + 1.0 / n_lm_layers)

    if DEBUG:
        fluid.layers.Print(normed_weights, first_n=1, summarize=10, message="normed_weights")
    splited_normed_weights = layers.split(normed_weights, n_lm_layers, dim=0)

    # compute the weighted, normalized LM activations
    pieces = []
    for w, t in zip(splited_normed_weights, lm_embeddings):
        pieces.append(t * w)
    sum_pieces = layers.sums(pieces)
    if DEBUG:
        fluid.layers.Print(sum_pieces, first_n=1, summarize=10, message="weighted_mebeddings")

    # scale the weighted sum by gamma
    gamma = layers.create_parameter([1], dtype="float32", name=name+"ELMo_gamma",
                                attr=fluid.ParamAttr(name=name+"ELMo_gamma",
                                                     initializer=fluid.initializer.Constant(1.0)))
    weighted_lm_layers = sum_pieces * gamma
    if DEBUG:
        fluid.layers.Print(gamma, first_n=1, summarize=10, message="gamma")
        fluid.layers.Print(weighted_lm_layers, first_n=1, summarize=10, message="weighted_mebeddings_multi_gamma")

    return weighted_lm_layers


def elmo_encoder(x_emb, elmo_l2_coef):
    #args modify

    lstm_outputs = []

    x_emb_r=fluid.layers.sequence_reverse(x_emb, name=None)

    fw_hiddens, fw_hiddens_ori = encoder_1(
        x_emb,
        vocab_size,
        emb_size,
        para_name='fw_',
        args=None)
    bw_hiddens, bw_hiddens_ori = encoder_1(
         x_emb_r,
         vocab_size,
         emb_size,
         para_name='bw_',
         args=None)

    num_layers = len(fw_hiddens_ori)
    token_embeddings = layers.concat(input=[x_emb, x_emb], axis=1)
    token_embeddings.stop_gradient = True
    concate_embeddings = [token_embeddings]
    for index in range(num_layers):
        #embedding = layers.concat(input = [fw_hiddens[index], bw_hiddens[index]], axis=1)
        embedding = layers.concat(input = [fw_hiddens_ori[index], bw_hiddens_ori[index]], axis=1)
        if modify == 1:
            embedding = dropout(embedding)
        embedding.stop_gradient=True
        concate_embeddings.append(embedding)
    weighted_meb = weight_layers(concate_embeddings, l2_coef=elmo_l2_coef)
    return weighted_meb
