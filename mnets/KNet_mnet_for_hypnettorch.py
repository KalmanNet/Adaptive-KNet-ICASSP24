#!/usr/bin/env python3
# Copyright 2019 Maria Cervera
# Modified by: 2023 Xiaoyong Ni 
"""
KalmanNet as main model for hypernetworks.
---------

An example usage is as a main model, where the main weights are initialized
and protected by a method such as EWC, and the context-modulation patterns of
the neurons are produced by an external hypernetwork.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from KalmanNet_nn import KalmanNetNN
from hypnettorch.mnets.mnet_interface import MainNetInterface
from hypnettorch.utils.torch_utils import init_params

class mKNet(nn.Module, MainNetInterface):
    """Implementation of KalmanNet as main model for hypernetworks.        
    """
    def __init__(self, config, SysModel, no_weights=False, 
                 init_weights=None, kaiming_rnn_init=False, 
                 context_mod_num_ts=-1,
                 context_mod_separate_layers_per_ts=False,**kwargs):

        nn.Module.__init__(self)
        MainNetInterface.__init__(self)
        ##########################################
        ### Parse or set context-mod arguments ###
        ##########################################

        if context_mod_num_ts != -1 and not self._context_mod_no_weights and \
                    not context_mod_separate_layers_per_ts:
                raise ValueError('When applying context-mod per timestep ' +
                    'while maintaining weights internally, option' +
                    '"context_mod_separate_layers_per_ts" must be set.')

        rem_kwargs = MainNetInterface._parse_context_mod_args(kwargs)
        if len(rem_kwargs) > 0:
            raise ValueError('Keyword arguments %s unknown.' % str(rem_kwargs))

        self._use_context_mod = kwargs['use_context_mod']
        self._context_mod_inputs = kwargs['context_mod_inputs']
        self._no_last_layer_context_mod = kwargs['no_last_layer_context_mod']
        self._context_mod_no_weights = kwargs['context_mod_no_weights']
        self._context_mod_post_activation = True
        self._context_mod_gain_offset = kwargs['context_mod_gain_offset']
        self._context_mod_gain_softplus = kwargs['context_mod_gain_softplus']

         # Context-mod options specific to RNNs
        self._context_mod_last_step = False
        self._context_mod_separate_layers_per_ts = -1 # different ts use the same CM 
            
        # More appropriate naming of option.
        self._context_mod_outputs = not self._no_last_layer_context_mod

        self._no_weights = no_weights # True if no internal weights(i.e. all generated externally by hypernetwork)
        # FIXME We have to specify this option even if
        # `context_mod_separate_layers_per_ts` is False (in order to set
        # sensible parameter shapes). However, the forward method can deal with
        # an arbitrary timestep length.
        self._context_mod_num_ts = context_mod_num_ts
        self._context_mod_separate_layers_per_ts = \
            context_mod_separate_layers_per_ts
        
        ######################
        ### Init KalmanNet ###
        ######################
        
        self.KalmanNet = KalmanNetNN()
        self.KalmanNet.NNBuild(SysModel, config)
        
        # Define fc and gru output layer shapes for later CM layer construction
        self.fc_outputlayers_shape = [self.KalmanNet.d_output_FC1, self.KalmanNet.d_output_FC2,\
                                self.KalmanNet.d_output_FC3, self.KalmanNet.d_output_FC4,\
                                self.KalmanNet.d_output_FC5, self.KalmanNet.d_output_FC6,\
                                self.KalmanNet.d_output_FC7]
        
        self.gru_outputlayers_shape = [self.KalmanNet.d_hidden_Q, self.KalmanNet.d_hidden_Sigma,\
                                       self.KalmanNet.d_hidden_S]
        
        # Define original KNet fc and gru layer shapes for later internal layer construction
        self.fc_layers_shape = [[self.KalmanNet.d_input_FC1, self.KalmanNet.d_output_FC1],[self.KalmanNet.d_input_FC2, self.KalmanNet.d_hidden_FC2, self.KalmanNet.d_output_FC2],[self.KalmanNet.d_input_FC3, self.KalmanNet.d_output_FC3],\
                                [self.KalmanNet.d_input_FC4, self.KalmanNet.d_output_FC4],[self.KalmanNet.d_input_FC5, self.KalmanNet.d_output_FC5],[self.KalmanNet.d_input_FC6, self.KalmanNet.d_output_FC6],[self.KalmanNet.d_input_FC7, self.KalmanNet.d_output_FC7]]
        
        self.gru_layers_shape = [self.KalmanNet.d_input_Q, self.KalmanNet.d_hidden_Q, self.KalmanNet.d_input_Sigma, self.KalmanNet.d_hidden_Sigma,\
                      self.KalmanNet.d_input_S, self.KalmanNet.d_hidden_S]

        ########################
        ### Init param lists ###
        ########################

        self._param_shapes = []
        self._param_shapes_meta = []

        self._internal_params = None if no_weights and self._context_mod_no_weights \
            else nn.ParameterList()
        self._hyper_shapes_learned = None \
            if not no_weights and not self._context_mod_no_weights else []
        self._hyper_shapes_learned_ref = None if self._hyper_shapes_learned \
            is None else []

        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        #################################################
        ### Define and initialize context mod weights ###
        #################################################

        # The context-mod layers apply to the output layers of original KNet ordered as:
        # FC 1-7: 2,4,6,8,10,12,14
        # GRU Q -> GRU sigma -> GRU S: 16,18,20

        self._context_mod_layers = nn.ModuleList() if self._use_context_mod \
            else None

        self._cm_gru_start_ind = 0
        self._num_fc_cm_layers = None
        if self._use_context_mod:
            cm_layer_inds = []
            cm_shapes = []

            # Gather sizes of all activation vectors within the network that
            # will be subject to context-modulation.
            if self._context_mod_inputs:
                self._cm_gru_start_ind += 1
                cm_shapes.append([SysModel.n]) # input size is SysModel.n

                # We reserve layer zero for input context-mod. Otherwise, there
                # is no layer zero.
                cm_layer_inds.append(0)
           
            self._cm_gru_start_ind += len(self.fc_outputlayers_shape)

            # We use odd numbers for actual layers and even number for all
            # context-mod layers.
            rem_cm_inds = range(2, 2*(len(self.fc_outputlayers_shape)+len(self.gru_outputlayers_shape))+1, 2)

            num_rec_cm_layers = len(self.gru_outputlayers_shape)
            self._num_rec_cm_layers = num_rec_cm_layers

            jj = 0
            # Add initial fully-connected context-mod layers.
            num_fc_cm_layers = len(self.fc_outputlayers_shape)
            for i in range(num_fc_cm_layers):
                cm_shapes.append([self.fc_outputlayers_shape[i]])
                cm_layer_inds.append(rem_cm_inds[jj])
                jj += 1

            # Add recurrent context-mod layers.
            for i in range(num_rec_cm_layers):
                if context_mod_num_ts != -1:
                    if context_mod_separate_layers_per_ts:
                        cm_rnn_shapes = [[self.gru_outputlayers_shape[i]]] * context_mod_num_ts
                    else:
                        # Only a single context-mod layer will be added, but we
                        # directly edit the correponding `param_shape` later.
                        assert self._context_mod_no_weights
                        cm_rnn_shapes = [[self.gru_outputlayers_shape[i]]]
                else:
                    cm_rnn_shapes = [[self.gru_outputlayers_shape[i]]]

                cm_shapes.extend(cm_rnn_shapes)
                cm_layer_inds.extend([rem_cm_inds[jj]] * len(cm_rnn_shapes))
                jj += 1

            self._add_context_mod_layers(cm_shapes, cm_layers=cm_layer_inds) # already included definition of _hyper_shapes_learned, _internal_params, etc.

            if context_mod_num_ts != -1 and not context_mod_separate_layers_per_ts:
            # In this case, there is only one context-mod layer for each
            # recurrent layer, but we want to have separate weights per
            # timestep.
            # Hence, we adapt the expected parameter shape, such that we
            # get a different set of weights per timestep. This will be
            # split into multiple weights that are succesively fed into the
            # same layer inside the forward method.
                for i in range(num_rec_cm_layers):
                    cmod_layer = \
                        self.context_mod_layers[self._cm_gru_start_ind+i]
                    # FIXME: what if T != T_test
                    cm_shapes_rnn = [[SysModel.T, *s] for s in \
                                        cmod_layer.param_shapes]

                    ps_ind = int(np.sum([ \
                        len(self.context_mod_layers[ii].param_shapes) \
                        for ii in range(self._cm_gru_start_ind+i)]))
                    self._param_shapes[ps_ind:ps_ind+len(cm_shapes_rnn)] = \
                        cm_shapes_rnn
                    assert self._hyper_shapes_learned is not None
                    self._hyper_shapes_learned[ \
                        ps_ind:ps_ind+len(cm_shapes_rnn)] = cm_shapes_rnn

        ########################
        ### Internal weights ###
        ########################
        # If no_weights is True, then the internal weights defined here are also generated by hypernetworks.

        # The original KNet weights are initialized in the following order:
        # FC 1-7: 1,3&5,7,9,11,13,15
        # GRU Q -> GRU sigma -> GRU S: 17,19,21

        def define_fc_layer_weights(FC, fc_layers, prev_dim, num_prev_layers):
            """Define the weights and shapes of the fully-connected layers.

            Args:
                FC(nn.Module): The module of fully-connected layer.
                fc_layers (list): The list of fully-connected layer dimensions.
                prev_dim (int): The output size of the previous layer.
                num_prev_layers (int): The number of upstream layers to the 
                    current one (a layer with its corresponding
                    context-mod layer(s) count as one layer). Count should
                    start at ``1``.

            Returns:
                (int): The output size of the last fully-connected layer
                considered here.
            """
            for i, n_fc in enumerate(fc_layers):
                s_w = [n_fc, prev_dim]
                s_b = [n_fc] 

                for j, s in enumerate([s_w, s_b]):
                    if s is None:
                        continue
                    
                    is_bias = True
                    if j % 2 == 0:
                        is_bias = False

                    if not self._no_weights:
                        self._internal_params.append(FC[2*i].bias if is_bias \
                                                     else FC[2*i].weight)
                        if is_bias:
                            self._layer_bias_vectors.append(self._internal_params[-1])
                        else:
                            self._layer_weight_tensors.append(self._internal_params[-1])
                    else:
                        self._hyper_shapes_learned.append(s)
                        self._hyper_shapes_learned_ref.append( \
                            len(self.param_shapes))

                    self._param_shapes.append(s)
                    self._param_shapes_meta.append({
                        'name': 'bias' if is_bias else 'weight',
                        'index': -1 if self._no_weights else \
                            len(self._internal_params)-1,
                        'layer': i * 2 + 1 + 2 * num_prev_layers, # Odd numbers
                    })

                num_prev_layers += 1

            return num_prev_layers

        ### Initial fully-connected layers.
        num_prev_layers = 0
        num_prev_layers = define_fc_layer_weights(self.KalmanNet.FC1, [self.fc_layers_shape[0][1]], self.fc_layers_shape[0][0], num_prev_layers)
        num_prev_layers = define_fc_layer_weights(self.KalmanNet.FC2, [self.fc_layers_shape[1][1], self.fc_layers_shape[1][2]], self.fc_layers_shape[1][0], \
                                                num_prev_layers)
        num_prev_layers = define_fc_layer_weights(self.KalmanNet.FC3, [self.fc_layers_shape[2][1]], self.fc_layers_shape[2][0], \
                                                num_prev_layers)
        num_prev_layers = define_fc_layer_weights(self.KalmanNet.FC4, [self.fc_layers_shape[3][1]], self.fc_layers_shape[3][0], \
                                                num_prev_layers)
        num_prev_layers = define_fc_layer_weights(self.KalmanNet.FC5, [self.fc_layers_shape[4][1]], self.fc_layers_shape[4][0], \
                                                num_prev_layers)
        num_prev_layers = define_fc_layer_weights(self.KalmanNet.FC6, [self.fc_layers_shape[5][1]], self.fc_layers_shape[5][0], \
                                                num_prev_layers)
        num_prev_layers = define_fc_layer_weights(self.KalmanNet.FC7, [self.fc_layers_shape[6][1]], self.fc_layers_shape[6][0], \
                                                num_prev_layers)
        
        print('fc_layers_pre', num_prev_layers) # should be 8
        ### Recurrent layers.
        def define_gru_layer_weights(gru, num_prev_layers):

            i = 0 # GRU Q,sigma,S all have only 1 layer.
            # Input-to-hidden
            s_w_ih = [gru.weight_ih_l0.shape[0], gru.weight_ih_l0.shape[1]]
            s_b_ih = [gru.bias_ih_l0.shape[0]] 

            # Hidden-to-hidden
            s_w_hh = [gru.weight_hh_l0.shape[0], gru.weight_hh_l0.shape[1]]
            s_b_hh = [gru.bias_hh_l0.shape[0]] 

            # Hidden-to-output. Note that GRU don't have this.
            s_w_ho = None
            s_b_ho = None

            for j, s in enumerate([s_w_ih, s_b_ih, s_w_hh, s_b_hh, s_w_ho,
                                s_b_ho]):
                if s is None:
                    continue

                is_bias = True
                if j % 2 == 0:
                    is_bias = False

                wtype = 'ih'
                if 2 <= j < 4:
                    wtype = 'hh'

                if not no_weights:
                    if is_bias:
                        self._internal_params.append(gru.bias_ih_l0 if wtype == 'ih' \
                                            else gru.bias_hh_l0)
                    else:
                        self._internal_params.append(gru.weight_ih_l0 if wtype == 'ih' \
                                            else gru.weight_hh_l0)
                        
                    if is_bias:
                        self._layer_bias_vectors.append(self._internal_params[-1])
                    else:
                        self._layer_weight_tensors.append(self._internal_params[-1])
                else:
                    self._hyper_shapes_learned.append(s)
                    self._hyper_shapes_learned_ref.append( \
                        len(self.param_shapes))

                self._param_shapes.append(s)
                self._param_shapes_meta.append({
                    'name': 'bias' if is_bias else 'weight',
                    'index': -1 if no_weights else len(self._internal_params)-1,
                    'layer': i * 2 + 1 + 2 * len(num_prev_layers), # Odd numbers
                    'info': wtype
                })

            num_prev_layers += 1
            return num_prev_layers

        rec_start = num_prev_layers
        num_prev_layers = define_gru_layer_weights(self.KalmanNet.GRU_Q, num_prev_layers)
        num_prev_layers = define_gru_layer_weights(self.KalmanNet.GRU_sigma, num_prev_layers)
        num_prev_layers = define_gru_layer_weights(self.KalmanNet.GRU_S, num_prev_layers) 

        print('total FC and GRU layer number', num_prev_layers) # should be 11  

        ### Initialize weights.(for internal weights)
        if init_weights is not None:
            assert self._internal_params is not None
            assert len(init_weights) == len(self.weights)
            for i in range(len(init_weights)):
                assert np.all(np.equal(list(init_weights[i].shape),
                                       self.weights[i].shape))
                self.weights[i].data = init_weights[i]
        else:
            rec_end = rec_start + 3 * 2 # 3 GRUs, 2 layers each (ih and hh).
            # Note, Pytorch applies a uniform init to its recurrent layers, as
            # defined here:
            # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py#L155
            for i in range(len(self._layer_weight_tensors)):
                if i >=rec_start and i < rec_end:
                    # Recurrent layer weights.
                    if kaiming_rnn_init:
                        init_params(self._layer_weight_tensors[i],
                            self._layer_bias_vectors[i])
                    else:
                        a = 1.0 / math.sqrt(self.gru_layers_shape[(i-rec_start) // 2])
                        nn.init.uniform_(self._layer_weight_tensors[i], -a, a)
                        
                        nn.init.uniform_(self._layer_bias_vectors[i], -a, a)
                else:
                    # FC layer weights.
                    init_params(self._layer_weight_tensors[i],
                        self._layer_bias_vectors[i])

        num_weights = MainNetInterface.shapes_to_num_weights(self._param_shapes)

        # Print information(e.g., the number of weights) during the construction of the network
        if self._use_context_mod:
            cm_num_weights =  \
                MainNetInterface.shapes_to_num_weights(cm_shapes)

        print('Creating KNet with %d weights' % num_weights
                + (' (including %d weights associated with-' % cm_num_weights
                    + 'context modulation)' if self._use_context_mod else '')
                + '.')

        self._is_properly_setup()

    
    def forward(self, init, y, weights=None, distilled_params=None, condition=None):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            init (torch.Tensor): The initial state of the Kalman filter.
                [batch size, m, 1]
            y (torch.Tensor): The input to the network, i.e. observations. 
                [sequence length, batch size, n]
            weights (list or dict): See argument ``weights`` of method
                :meth:`mnets.mlp.MLP.forward`.
            condition (optional, int): If provided, then this argument will be
                passed as argument ``ckpt_id`` to the method
                :meth:`utils.context_mod_layer.ContextModLayer.forward`.

        Returns:
            - **output** (torch.Tensor): The output of the network.
        """
        assert distilled_params is None

        if ((not self._use_context_mod and self._no_weights) or \
                (self._no_weights or self._context_mod_no_weights)) and \
                weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        #######################
        ### Extract weights ###
        #######################
        # Extract which weights should be used.
        int_weights, cm_weights = self.split_weights(weights)

        ### Split context-mod weights per context-mod layer.
        cm_inputs_weights, cm_fc_layer_weights, cm_rec_layer_weights, \
            n_cm_rec, cmod_cond = self.split_cm_weights(
                cm_weights, condition, num_ts=y.shape[0])

        ### Extract internal weights.
        fc_w_weights, fc_b_weights, rec_weights = self.split_internal_weights(int_weights)

        ###########################
        ### Forward Computation ###
        ###########################       
        # input first go through a CM layer
        cm_offset = 0
        if self._use_context_mod and self._context_mod_inputs:
            cm_offset += 1
            # Apply context modulation in the inputs.
            y = self._context_mod_layers[0].forward(y,
                weights=cm_inputs_weights, ckpt_id=cmod_cond, bs_dim=1)
        
        # transpose y to fit KalmanNet
        y = torch.permute(y,(1,2,0)) #[batch size, n, sequence length]
        self.KalmanNet.batch_size = y.shape[0]
        # Init Hidden State each time we start from t=0
        self.KalmanNet.init_hidden_KNet()
        # Init Sequence
        self.KalmanNet.InitSequence(init, y.shape[2])
        # Init output tensor [batch size, m, sequence length]
        x_out_training_batch = torch.zeros([y.shape[0], init.shape[1], y.shape[2]]).to(y.device)            
        # t=0:T-1, do KalmanNet forward
        for t in range(y.shape[2]): 
            yt_training_batch = torch.unsqueeze(y[:, :, t],2)  
            # Compute Priors
            self.KalmanNet.step_prior()
            # Compute Kalman Gain
            self.KalmanNet.step_KGain_est(yt_training_batch)
            # Innovation
            dy = yt_training_batch - self.KalmanNet.m1y # [batch_size, n, 1]
            # Compute the 1-st posterior moment
            INOV = torch.bmm(self.KalmanNet.KGain, dy)
            self.KalmanNet.m1x_posterior_previous = self.KalmanNet.m1x_posterior
            self.KalmanNet.m1x_posterior = self.KalmanNet.m1x_prior + INOV

            #self.state_process_posterior_0 = self.state_process_prior_0
            self.KalmanNet.m1x_prior_previous = self.KalmanNet.m1x_prior

            # update y_prev
            self.KalmanNet.y_previous = yt_training_batch

            x_out_training_batch[:, :, t] = torch.squeeze(self.KalmanNet.m1x_posterior) 

        # transpose x_out_training_batch to fit hypernet
        x_out_training_batch = torch.permute(x_out_training_batch,(2,0,1)) #[sequence length, batch size, m]

        return x_out_training_batch

    def split_weights(self, weights):
        """Split weights into internal and context-mod weights.
        Extract which weights should be used,  I.e., are we using internally
        maintained weights or externally given ones or are we even mixing
        between these groups.
        Args:
            weights (torch.Tensor): All weights.
        Returns:
            (Tuple): Where the tuple contains:
            - **int_weights**: The internal weights.
            - **cm_weights**: The context-mod weights.
        """
        n_cm = self._num_context_mod_shapes()

        # Make sure cm_weights are either `None` or have the correct dimensions.
        if weights is None:
            weights = self.weights

            if self._use_context_mod:
                cm_weights = weights[:n_cm]
                int_weights = weights[n_cm:]
            else:
                cm_weights = None
                int_weights = weights
        else:
            int_weights = None
            cm_weights = None

            if isinstance(weights, dict):
                assert 'internal_weights' in weights.keys() or \
                    'mod_weights' in weights.keys()
                if 'internal_weights' in weights.keys():
                    int_weights = weights['internal_weights']
                if 'mod_weights' in weights.keys():
                    cm_weights = weights['mod_weights']
            else:
                if self._use_context_mod and \
                        len(weights) == n_cm:
                    cm_weights = weights
                else:
                    assert len(weights) == len(self.param_shapes)
                    if self._use_context_mod:
                        cm_weights = weights[:n_cm]
                        int_weights = weights[n_cm:]
                    else:
                        int_weights = weights

            if self._use_context_mod and cm_weights is None:
                if self._context_mod_no_weights:
                    raise Exception('Network was generated without weights ' +
                        'for context-mod layers. Hence, they must be passed ' +
                        'via the "weights" option.')
                cm_weights = self.weights[:n_cm]
            if int_weights is None:
                if self._no_weights:
                    raise Exception('Network was generated without internal ' +
                        'weights. Hence, they must be passed via the ' +
                        '"weights" option.')
                if self._context_mod_no_weights:
                    int_weights = self.weights
                else:
                    int_weights = self.weights[n_cm:]

            # Note, context-mod weights might have different shapes, as they
            # may be parametrized on a per-sample basis.
            if self._use_context_mod:
                assert len(cm_weights) == n_cm
            int_shapes = self.param_shapes[n_cm:]
            assert len(int_weights) == len(int_shapes)
            for i, s in enumerate(int_shapes):
                assert np.all(np.equal(s, list(int_weights[i].shape)))

        return int_weights, cm_weights
    
    def split_cm_weights(self, cm_weights, condition, num_ts=0):
        """Split context-mod weights per context-mod layer.
        Args:
            cm_weights (torch.Tensor): All context modulation weights.
            condition (optional, int): If provided, then this argument will be
                passed as argument ``ckpt_id`` to the method
                :meth:`utils.context_mod_layer.ContextModLayer.forward`.
            num_ts (int): The length of the sequences.
        Returns:
            (Tuple): Where the tuple contains:
            - **cm_inputs_weights**: The cm input weights.
            - **cm_fc_layer_weights**: The cm FC1-7 weights.
            - **cm_rec_layer_weights**: The cm recurrent weights.            
            - **n_cm_rec**: The number of recurrent cm layers.
            - **cmod_cond**: The context-mod condition.
        """

        n_cm_rec = -1
        cm_inputs_weights = None
        cm_fc_layer_weights = None       
        cm_rec_layer_weights = None
        if cm_weights is not None:
            if self._context_mod_num_ts != -1 and \
                    self._context_mod_separate_layers_per_ts:
                assert num_ts <= self._context_mod_num_ts

            # Note, an mnet layer might contain multiple context-mod layers
            # (a recurrent layer can have a separate context-mod layer per
            # timestep).
            cm_fc_layer_weights = []
            cm_rec_layer_weights = [[] for _ in range(self._num_rec_cm_layers)]

            # Number of cm-layers per recurrent layer.
            n_cm_per_rec = self._context_mod_num_ts if \
                self._context_mod_num_ts != -1 and \
                    self._context_mod_separate_layers_per_ts else 1
            n_cm_rec = n_cm_per_rec * self._num_rec_cm_layers

            cm_start = 0
            for i, cm_layer in enumerate(self.context_mod_layers):
                cm_end = cm_start + len(cm_layer.param_shapes)

                if i == 0 and self._context_mod_inputs:
                    cm_inputs_weights = cm_weights[cm_start:cm_end]
                elif i < self._cm_gru_start_ind:
                    cm_fc_layer_weights.append(cm_weights[cm_start:cm_end])
                else:
                    # Index of recurrent layer.
                    i_r = (i-self._cm_gru_start_ind) // n_cm_per_rec
                    cm_rec_layer_weights[i_r].append( \
                        cm_weights[cm_start:cm_end])
                cm_start = cm_end

            # We need to split the context-mod weights in the following case,
            # as they are currently just stacked on top of each other.
            if self._context_mod_num_ts != -1 and \
                    not self._context_mod_separate_layers_per_ts:
                for i, cm_w_list in enumerate(cm_rec_layer_weights):
                    assert len(cm_w_list) == 1

                    cm_rnn_weights = cm_w_list[0]
                    cm_rnn_layer = self.context_mod_layers[ \
                        self._cm_gru_start_ind+i]

                    assert len(cm_rnn_weights) == len(cm_rnn_layer.param_shapes)
                    # The first dimension are the weights of this layer per
                    # timestep.
                    num_ts_cm = -1
                    for j, s in enumerate(cm_rnn_layer.param_shapes):
                        assert len(cm_rnn_weights[j].shape) == len(s) + 1
                        if j == 0:
                            num_ts_cm = cm_rnn_weights[j].shape[0]
                        else:
                            assert num_ts_cm == cm_rnn_weights[j].shape[0]
                    assert num_ts <= num_ts_cm

                    cm_w_chunked = [None] * len(cm_rnn_weights)
                    for j, cm_w in enumerate(cm_rnn_weights):
                        cm_w_chunked[j] = torch.chunk(cm_w, num_ts_cm, dim=0)

                    # Now we gather all these chunks to assemble the weights
                    # needed per timestep (as if
                    # `_context_mod_separate_layers_per_t` were True).
                    cm_w_list = []
                    for j in range(num_ts_cm):
                        tmp_list = []
                        for chunk in cm_w_chunked:
                            tmp_list.append(chunk[j].squeeze(dim=0))
                        cm_w_list.append(tmp_list)
                    cm_rec_layer_weights[i] = cm_w_list

            # Note, the last layer does not necessarily have context-mod
            # (depending on `self._context_mod_outputs`).
            if len(cm_rec_layer_weights) < len(self.gru_outputlayers_shape):
                cm_rec_layer_weights.append(None)
            if len(cm_fc_layer_weights) < len(self.fc_outputlayers_shape):
                cm_fc_layer_weights.append(None)


        #######################
        ### Parse condition ###
        #######################
        cmod_cond = None
        if condition is not None:
            assert isinstance(condition, int)
            cmod_cond = condition

            # Note, the cm layer will ignore the cmod condition if weights
            # are passed.
            # FIXME Find a more elegant solution.
            cm_inputs_weights = None
            cm_fc_layer_weights = [None] * len(cm_fc_layer_weights)
            cm_rec_layer_weights = [[None] * len(cm_ws) for cm_ws in \
                                    cm_rec_layer_weights]
            
        return cm_inputs_weights, cm_fc_layer_weights, cm_rec_layer_weights,\
              n_cm_rec, cmod_cond

    def split_internal_weights(self, int_weights):
        """Split internal weights per layer.
        Args:
            int_weights (torch.Tensor): All internal weights.
        Returns:
            (Tuple): Where the tuple contains:
            - **fc_w_weights**: FC 1-7 w weights.
            - **fc_b_weights**: FC 1-7 b weights.
            - **rec_weights**: The recurrent weights.

        """
        n_cm = self._num_context_mod_shapes() # defined inside mnet_interface

        int_meta = self.param_shapes_meta[n_cm:]
        assert len(int_meta) == len(int_weights)
        fc_w_weights = []
        fc_b_weights = []
        rec_weights =[[] for _ in range(len(self.gru_outputlayers_shape))]

        # Number of pre-fc weights in total.
        n_fc_pre = 8 * 2 # FC 1-7 (FC 2 has 2 layers), *2 for w and b

        # Number of weights per recurrent layer.
        n_rw = 4 # 4 for GRU

        for i, w in enumerate(int_weights):
            if i < n_fc_pre: # fc weights
                if int_meta[i]['name'] == 'weight':
                    fc_w_weights.append(w)
                else:
                    assert int_meta[i]['name'] == 'bias'
                    fc_b_weights.append(w)
            else: # recurrent w
                r_ind = (i - n_fc_pre) // n_rw
                rec_weights[r_ind].append(w)

        return fc_w_weights, fc_b_weights, rec_weights


if __name__ == '__main__':
    pass
    # mKNet.__init__(SysModel, no_weights=False, 
    #              init_weights=None, kaiming_rnn_init=False, **kwargs)

