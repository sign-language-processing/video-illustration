
"""
Util functions based on prompt-to-prompt.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import  abc
from einops import rearrange
from typing import Optional, Union, Tuple, List, Callable, Dict
import math

from torchvision.utils import save_image
OUT_INDEX = 4

def enhance_tensor(tensor: torch.Tensor, contrast_factor: float = 1.67) -> torch.Tensor:
    """ Compute the attention map contrasting. """
    adjusted_tensor = (tensor - tensor.mean(dim=-1)) * contrast_factor + tensor.mean(dim=-1)
    return adjusted_tensor

def compute_scaled_dot_product_attention(Q, K, V, edit_map=False, is_cross=False, contrast_strength=1.0):
    """ Compute the scale dot product attention, potentially with our contrasting operation. """
    attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)
    if edit_map and not is_cross:
        attn_weight[OUT_INDEX] = torch.stack([
            torch.clip(enhance_tensor(attn_weight[OUT_INDEX][head_idx], contrast_factor=contrast_strength),
                       min=0.0, max=1.0)
            for head_idx in range(attn_weight.shape[1])
        ])
    return attn_weight @ V, attn_weight

# TO DO: 
def register_attention_control_ICL(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            x = hidden_states
            context = encoder_hidden_states
            mask = attention_mask
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            is_cross = context is not None

            context = context if is_cross else x
            
            # k, q, v = controller(hidden_states, is_cross, place_in_unet)


            # k = hidden_states[first_dim// 2:][1]
            # q = hidden_states[first_dim// 2:][2]
            # v = hidden_states[first_dim// 2:][3]

            queries = hidden_states
            keys = hidden_states
            values = hidden_states

            keys[OUT_INDEX] = hidden_states[5]
            queries[OUT_INDEX] = hidden_states[6]
            values[OUT_INDEX] = hidden_states[7]

            
            if is_cross:    
                q = self.to_q(x)
                k = self.to_k(context)
                v = self.to_v(context)
            else:
                q = self.to_q(queries)
                k = self.to_k(keys)
                v = self.to_v(values)
            
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)
            
            tau = 0.4
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
                
            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            if not is_cross:
                sim[OUT_INDEX] /= tau

            attn = sim.softmax(dim=-1)

            if not is_cross:
                # attn_modified = self.batch_to_head_dim(attn)
                # query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                attn_modified = attn.detach().clone()
                attn_modified = attn_modified.view(batch_size, self.heads, *attn.size()[-2:])
                attn_modified[OUT_INDEX] = torch.stack([torch.clip(enhance_tensor(attn_modified[OUT_INDEX][head_idx]),
                                                           min=0.0, max=1.0) for head_idx in range(attn_modified.shape[1])])
                # adjusted_tensor = (tensor - tensor.mean(dim=-1)) * contrast_factor + tensor.mean(dim=-1)
                attn = attn_modified.view(batch_size*self.heads, *attn.size()[-2:])
                
                # attn_modified = self.head_to_batch_dim(attn_modified)
            # if is_cross:
            #     ssss = []
            # attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)
            return to_out(out)
                     

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        # print(net_.__class__.__name__)
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    # print(sub_nets)
    for net in sub_nets:
        if "down" in net[0]:
            # cross_att_count += register_recr(net[1], 0, "down")
            pass
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            # cross_att_count += register_recr(net[1], 0, "mid")
            pass

    controller.num_att_layers = cross_att_count



class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, hidden_states, is_cross: bool, place_in_unet: str, custom_latents):
        raise NotImplementedError
    
    def __call__(self, hidden_states, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = hidden_states.shape[0]
            k, q, v = self.forward(hidden_states, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # self.between_steps()
        return  k, q, v 
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.LOW_RESOURCE = False

        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, hidden_states, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"

        return hidden_states

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):

                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

class SelfAttentionControlICL(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    
    def forward(self, hidden_states, is_cross: bool, place_in_unet: str):
        super(SelfAttentionControlICL, self).forward(hidden_states, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):

            first_dim = hidden_states.shape[0]

            k = hidden_states[first_dim// 2:][1]
            q = hidden_states[first_dim// 2:][2]
            v = hidden_states[first_dim// 2:][3]
            #(latents, latents_k, latents_q, latents_v)

        return k, q, v
    
    def __init__(self, num_steps: int,
                 self_replace_steps: Union[float, Tuple[float, float]]):
        super(SelfAttentionControlICL, self).__init__()
        # self.batch_size = len(prompts)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])