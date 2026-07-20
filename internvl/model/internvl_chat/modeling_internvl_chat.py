# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from internvl.model.llama.modeling_llama import LlamaForCausalLM
from internvl.model.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel
import time
import torch
import os,json
import numpy as np
from causalUtils.anchors import generate_custom_anchors, anchors_to_token_mask
from causalUtils.predict import CausalInferencePredictor
logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


def get_attention_rank(visual_token_index, attentions):

    # assert visual_token_index.shape[0] == 1 # batchsize = 1
    # visual_token_index = visual_token_index.view(-1).nonzero()
    visual_start_index, visual_end_index = visual_token_index[0], visual_token_index[-1]

    attentions  = [torch.stack(attention, dim=1) for attention in attentions] # [n l heads tokens, tokens]


    visual_token_importance = 0.0
    for i, attn in enumerate(attentions):
        if i == 0:
            visual_token_importance += attn[0].sum(dim=0).sum(dim=0)[visual_end_index+1:, visual_start_index:visual_end_index+1].sum(dim=0)
        else:
            visual_token_importance += attn[0].sum(dim=0).sum(dim=0)[0:1, visual_start_index:visual_end_index+1].sum(dim=0)

    return visual_token_importance



class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None):
        super().__init__(config)
        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        
        if vision_model is not None:
            self.vision_model = vision_model
        else:   # 从这里进入
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):    # 图片特征提取
        if self.select_layer == -1:
            vit_embeds = self.vision_model(     # 对图像进行patch切分和特征提取，patch大小为32*32=1024个
                pixel_values=pixel_values,  # 5, 3, 448, 448
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]

        vit_embeds = vit_embeds[:, 1:, :]   # 5, 1025, 1024 -> 5 1024 1024

        h = w = int(vit_embeds.shape[1] ** 0.5) # 1024*0.5=32
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)  # 5 1024 1024 -> 5 32 32 1024
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio) # 5 32 32 1024 -> 5, 16, 16, 4096
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])  # 5, 16, 16, 4096 -> 5, 256, 4096
        vit_embeds = self.mlp1(vit_embeds) # 5, 256, 4096 -> 5, 256, 2048  这里再做“翻译”工作了，将图像token翻译成LLM能理解的token
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses
# ============ MODIFY START ============
    # 修改原因：需要加载Visual CoT数据集，新增VisualCoT的加载方法。对于其它数据集用else处理掉
    # 修改人：Taoyu Qian
    # 修改时间：2026-01-27
    def causalTrain(self, tokenizer, pixel_values, questions, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False, large_model=False):
        # for i in range(len(question)):
        if history is None and pixel_values is not None and '<image>' not in questions:
            reasonCausal = '<image>\n' + questions[0]    # 因
            resultCausal = '<image>\n' + questions[1]    # 果
            question_ids = [reasonCausal, resultCausal]

        question_ids_reason = tokenizer(questions[0], return_tensors='pt') # 先将问题编码
        question_ids_result = tokenizer(questions[1], return_tensors='pt')
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)   # pixel_values shape: 5, 3, 448, 448

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)   # 图像占位id
        self.img_context_token_id = img_context_token_id

        template_reason = get_conv_template(self.template)
        template_reason.system_message = self.system_message # 特殊提示词
        template_result = get_conv_template(self.template) 
        template_result.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template_reason.sep)


        history = [] if history is None else history
        # for (old_question, old_answer) in history:
        #     template.append_message(template.roles[0], old_question)  # 将历史记录append进template
        #     template.append_message(template.roles[1], old_answer)
        template_reason.append_message(template_reason.roles[0], reasonCausal)
        template_reason.append_message(template_reason.roles[1], None)
        query_reason = template_reason.get_prompt()   # 你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型
        template_result.append_message(template_result.roles[0], resultCausal)
        template_result.append_message(template_result.roles[1], None)
        query_result = template_result.get_prompt()   # 你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query_reason = query_reason.replace('<image>', image_tokens, 1)
            query_result = query_result.replace('<image>', image_tokens, 1)

        model_inputs_reason = tokenizer(query_reason, return_tensors='pt')    # 将query编码为token，大小为1340
        model_inputs_result = tokenizer(query_result, return_tensors='pt')    # 将query编码为token，大小为1340
        input_ids_reason = model_inputs_reason['input_ids'].cuda()    # 待输入token
        input_ids_result = model_inputs_result['input_ids'].cuda()    # 待输入token
        attention_mask_reason = model_inputs_reason['attention_mask'].cuda()
        attention_mask_result = model_inputs_result['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id

        # 计算viusal token的起始位置
        visual_token_index_reason = (input_ids_reason == self.img_context_token_id)
        visual_token_index_reason = visual_token_index_reason.view(-1).nonzero()
        visual_start_index_reason, visual_end_index_reason = visual_token_index_reason[0], visual_token_index_reason[-1] # 用于计算viusal token的起始位置
        visual_token_index_result = (input_ids_result == self.img_context_token_id)
        visual_token_index_result = visual_token_index_result.view(-1).nonzero()
        visual_start_index_result, visual_end_index_result = visual_token_index_result[0], visual_token_index_result[-1] # 用于计算viusal token的起始位置

        if large_model:
            generation_config["visual_token_index"] = (visual_start_index, visual_end_index)
            assert (visual_end_index - visual_start_index + 1) == generation_config["visual_token_importance"].shape[0]
        else:
            generation_config['consistency_config']["visual_token_index"] = (visual_start_index_reason, visual_end_index_reason, visual_start_index_result, visual_end_index_result)

        if not large_model:
            input_ids = [input_ids_reason, input_ids_result]
            attention_mask = [attention_mask_reason, attention_mask_result]
            _causalTrainGenerate = self.causalTrainGenerate(
                pixel_values=pixel_values,  # 5,3,448,448
                input_ids=input_ids,
                attention_mask=attention_mask,
                large_model=large_model,
                **generation_config
            )

        return _causalTrainGenerate[0], _causalTrainGenerate[1], question_ids_reason['input_ids'], question_ids_result['input_ids']

            # response = tokenizer.batch_decode(generation_output['sequences'], skip_special_tokens=True)[0]
            # response = response.split(template.sep)[0].strip()
            # history.append((question, response))
    
            # if return_history:
            #     return response, history
            # else:
            #     query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            #     query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            #     if verbose:
            #         print(query_to_print, response)
            #     return response, generation_output.scores, consistency_score, visual_token_importance

        # else:
        #     generation_output = self.generate(
        #         pixel_values=pixel_values,
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         large_model=large_model,
        #         **generation_config
        #     )

        #     response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        #     response = response.split(template.sep)[0].strip()
        #     history.append((question, response))
        #     if return_history:
        #         return response, history
        #     else:
        #         query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
        #         query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
        #         if verbose:
        #             print(query_to_print, response)
        #         return response

    def causalTest(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
                num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
                verbose=False, large_model=False):
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        question_ids_test = tokenizer(question, return_tensors='pt') # 先将问题编码
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)   # pixel_values shape: 5, 3, 448, 448

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        # for (old_question, old_answer) in history:
        #     template.append_message(template.roles[0], old_question)
        #     template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')    # 将query编码为token，大小为1340
        input_ids = model_inputs['input_ids'].cuda()    # 待输入token
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id

        visual_token_index = (input_ids == self.img_context_token_id)

        visual_token_index = visual_token_index.view(-1).nonzero()  
        visual_start_index, visual_end_index = visual_token_index[0], visual_token_index[-1]

    
        if large_model:
            generation_config["visual_token_index"] = (visual_start_index, visual_end_index)
            assert (visual_end_index - visual_start_index + 1) == generation_config["visual_token_importance"].shape[0]
        else:
            generation_config['consistency_config']["visual_token_index"] = (visual_start_index, visual_end_index)

        if not large_model:
            fin_tokens = self.causalTestGenerate(
                pixel_values=pixel_values,  # 5,3,448,448
                input_ids=input_ids,
                attention_mask=attention_mask,
                large_model=large_model,
                question_ids_test = question_ids_test,
                **generation_config
            )

            return fin_tokens
        
            # response = tokenizer.batch_decode(generation_output['sequences'], skip_special_tokens=True)[0]
            # response = response.split(template.sep)[0].strip()
            # history.append((question, response))
    
            # if return_history:
            #     return response, history
            # else:
            #     query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            #     query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            #     if verbose:
            #         print(query_to_print, response)
            #     return response, generation_output.scores, consistency_score, visual_token_importance

        else:
            generation_output = self.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                large_model=large_model,
                **generation_config
            )

            response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
            response = response.split(template.sep)[0].strip()
            history.append((question, response))
            if return_history:
                return response, history
            else:
                query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
                query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
                if verbose:
                    print(query_to_print, response)
                return response
# ============ MODIFY END ============


    def chat(self, fin_tokens, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False, large_model=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)   # pixel_values shape: 5, 3, 448, 448

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
  
        model_inputs = tokenizer(query, return_tensors='pt')    # 将query编码为token，大小为1340
        input_ids = model_inputs['input_ids'].cuda()    # 待输入token
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id

        visual_token_index = (input_ids == self.img_context_token_id)

        visual_token_index = visual_token_index.view(-1).nonzero()  
        visual_start_index, visual_end_index = visual_token_index[0], visual_token_index[-1]

       
        if large_model:
            generation_config["visual_token_index"] = (visual_start_index, visual_end_index)
            assert (visual_end_index - visual_start_index + 1) == len(generation_config["visual_token_importance"])
        else:
            generation_config['consistency_config']["visual_token_index"] = (visual_start_index, visual_end_index)
        
        #if not large_model:
            
        #     generation_output, consistency_score, visual_token_importance = self.generate(
        #         pixel_values=pixel_values,  # 5,3,448,448
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         large_model=large_model,
        #         **generation_config
        #     )

            # response = tokenizer.batch_decode(generation_output['sequences'], skip_special_tokens=True)[0]
            # response = response.split(template.sep)[0].strip()
            # history.append((question, response))
    
            # if return_history:
            #     return response, history
            # else:
            #     query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            #     query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            #     if verbose:
            #         print(query_to_print, response)
            #     return response, generation_output.scores, consistency_score, visual_token_importance

        # else:
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            large_model=large_model,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        # import pdb
        # pdb.set_trace()
        print("🤖："+str(response))
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response



    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,   # 图像像素点 5,3,448,448
            input_ids: Optional[torch.FloatTensor] = None,  # 1340 待输入token，已嵌入text，未嵌入image
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            large_model: Optional[bool] = False,
            **generate_kwargs,
    ) -> torch.LongTensor:
        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values) # 图像特征提取 5,3,448,448->5,256,2048   256应该是16*16的patch，2048是embedding维度即信息量

            # average_vit_embeds = torch.mean(vit_embeds, dim=2)  # 对第三维度求平均，测试用，结束后删除  qtyqtyqty
            # normalized_embeds = torch.nn.functional.normalize(average_vit_embeds, p=2, dim=1) # 归一化处理，不然太乱了看不清楚  qtyqtyqty

            input_embeds = self.language_model.get_input_embeddings()(input_ids)    # 1 1340 -> 1 1340 2048     2048应该是embedding维度，即单个token的信息量
            B, N, C = input_embeds.shape    # 1 1340 2048
            input_embeds = input_embeds.reshape(B * N, C)   # 1340 2048
            input_ids = input_ids.reshape(B * N)    # 1340
            selected = (input_ids == self.img_context_token_id) # 将非文本位置设为True，已有文本位置为False
            torch.set_printoptions(threshold=float('inf'))
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)  # 将图片token嵌入进input_input_embeds中 vit_embeds.reshape(-1, C)后大小为1280*2048  -1是 PyTorch 中的一个特殊值，表示自动计算该维度的大小，保持总元素数不变
            input_embeds = input_embeds.reshape(B, N, C)    # 1 1340 2048   input_embeds为最后要的嵌入token
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)   



        if not large_model:
            consistency_generate_kwargs = generate_kwargs.pop('consistency_config')
            generate_kwargs['visual_token_index'] = consistency_generate_kwargs['visual_token_index']
            outputs = self.language_model.generate(     # self.language_model为InternLM2ForCausalLM
                inputs_embeds=input_embeds, # 1340, 2048
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_cache=True,
                **generate_kwargs,
            )

            visual_token_importance = outputs.aggregated_viusal_token_attention
            consistency_generate_kwargs['visual_token_importance'] = visual_token_importance

            new_input_ids_ = outputs['sequences'][0]
            new_token_num = outputs['sequences'].shape[-1]
            new_input_embedding = torch.concatenate((input_embeds, self.language_model.get_input_embeddings()(new_input_ids_).unsqueeze(0)), dim=1)
            new_attention_mask = torch.concatenate((attention_mask, torch.ones((1, new_input_ids_.shape[0]), device=attention_mask.device, dtype=attention_mask.dtype)), dim=-1)
            new_input_ids = torch.concatenate((input_ids, new_input_ids_), dim=-1)
            consistency_generate_kwargs['inputs_embeds'] = new_input_embedding
            consistency_generate_kwargs['attention_mask'] = new_attention_mask
            consistency_generate_kwargs['output_scores'] = False
            consistency_generate_kwargs['output_attentions'] = False
            consistency_generate_kwargs = self.language_model._get_initial_cache_position(new_input_ids, consistency_generate_kwargs)

            model_inputs = self.language_model.prepare_inputs_for_generation(new_input_ids,  **consistency_generate_kwargs)
            consistency_output = self.language_model.forward(**model_inputs, return_dict=True)
            consistency_score = torch.gather(consistency_output['logits'][:, -new_token_num-1:-1, :].softmax(dim=-1), index=new_input_ids_[None, :, None], dim=-1)

            consistency_score = torch.prod(consistency_score)


            return outputs, consistency_score, visual_token_importance


        
        else:
            return self.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_cache=True,
                **generate_kwargs,
            )


    @torch.no_grad()
    def causalTrainGenerate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,   # 图像像素点 5,3,448,448
            input_ids: Optional[torch.FloatTensor] = None,  # 1340 待输入token，已嵌入text，未嵌入image
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            large_model: Optional[bool] = False,
            **generate_kwargs,
    ) -> torch.LongTensor:
        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values) # 图像特征提取 5,3,448,448->5,256,2048   256应该是16*16的patch，2048是embedding维度即信息量

            image_split_num = vit_embeds.shape[0]   # 当前图像被分割为多少张
            # average_vit_embeds = torch.mean(vit_embeds, dim=2)  # 对第三维度求平均，测试用，结束后删除  qtyqtyqty
            # normalized_embeds = torch.nn.functional.normalize(average_vit_embeds, p=2, dim=1) # 归一化处理，不然太乱了看不清楚  qtyqtyqty

            # reason处理
            input_embeds_reason = self.language_model.get_input_embeddings()(input_ids[0])    # 1 1340 -> 1 1340 2048     2048应该是embedding维度，即单个token的信息量
            B_reason, N_reason, C_reason = input_embeds_reason.shape    # 1 1340 2048
            input_embeds_reason = input_embeds_reason.reshape(B_reason * N_reason, C_reason)   # 1340 2048
            input_ids_reason = input_ids[0].reshape(B_reason * N_reason)    # 1340
            selected_reason = (input_ids_reason == self.img_context_token_id) # 将非文本位置设为True，已有文本位置为False
            torch.set_printoptions(threshold=float('inf'))
            assert selected_reason.sum() != 0

            input_embeds_reason[selected_reason] = vit_embeds.reshape(-1, C_reason).to(input_embeds_reason.device)  # 将图片token嵌入进input_input_embeds中 vit_embeds.reshape(-1, C)后大小为1280*2048  -1是 PyTorch 中的一个特殊值，表示自动计算该维度的大小，保持总元素数不变
            input_embeds_reason = input_embeds_reason.reshape(B_reason, N_reason, C_reason)    # 1 1340 2048   input_embeds为最后要的嵌入token
            # result 处理
            input_embeds_result = self.language_model.get_input_embeddings()(input_ids[1])    # 1 1340 -> 1 1340 2048     2048应该是embedding维度，即单个token的信息量
            B_result, N_result, C_result = input_embeds_result.shape    # 1 1340 2048
            input_embeds_result = input_embeds_result.reshape(B_result * N_result, C_result)   # 1340 2048
            input_ids_result = input_ids[1].reshape(B_result * N_result)    # 1340
            selected_result = (input_ids_result == self.img_context_token_id) # 将非文本位置设为True，已有文本位置为False
            torch.set_printoptions(threshold=float('inf'))
            assert selected_result.sum() != 0
            input_embeds_result[selected_result] = vit_embeds.reshape(-1, C_result).to(input_embeds_result.device)  # 将图片token嵌入进input_input_embeds中 vit_embeds.reshape(-1, C)后大小为1280*2048  -1是 PyTorch 中的一个特殊值，表示自动计算该维度的大小，保持总元素数不变
            input_embeds_result = input_embeds_result.reshape(B_result, N_result, C_result)    # 1 1340 2048   input_embeds为最后要的嵌入token
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)   



        if not large_model:
            consistency_generate_kwargs = generate_kwargs.pop('consistency_config')
            generate_kwargs['visual_token_index'] = consistency_generate_kwargs['visual_token_index']
            outputs_reason = self.language_model.generate(     # self.language_model为InternLM2ForCausalLM
                inputs_embeds=input_embeds_reason, # 1340, 2048
                attention_mask=attention_mask[0],
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_cache=True,
                **generate_kwargs,
            )
            outputs_result = self.language_model.generate(     # self.language_model为InternLM2ForCausalLM
                inputs_embeds=input_embeds_result, # 1340, 2048
                attention_mask=attention_mask[1],
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_cache=True,
                **generate_kwargs,
            )

            # 获取SGL聚合后结果
            visual_token_importance_reason = outputs_reason.aggregated_viusal_token_attention
            visual_token_importance_result = outputs_result.aggregated_viusal_token_attention
            visual_token_importance = [visual_token_importance_reason, visual_token_importance_result]
            consistency_generate_kwargs['visual_token_importance'] = visual_token_importance   

            visual_token_importance_reason = visual_token_importance_reason.unsqueeze(0)
            _, reasonLen = visual_token_importance_reason.shape
            start_idx_reason = int(reasonLen * (1 - 1/image_split_num))      # 计算截取初始位置 
            visual_token_importance_reason = visual_token_importance_reason[:, start_idx_reason:]  # 截取完整的图片的位置

            visual_token_importance_result = visual_token_importance_result.unsqueeze(0)
            _, resultLen = visual_token_importance_result.shape
            start_idx_result = int(resultLen * (1 - 1/image_split_num))      # 计算截取初始位置 
            visual_token_importance_result = visual_token_importance_result[:, start_idx_result:]
            # 将结果写入jsonl文件
            return [visual_token_importance_reason, visual_token_importance_result]
            # write_jsonl_train([visual_token_importance_reason, visual_token_importance_result], 1) # 因->果
            # write_jsonl_train([visual_token_importance_result, visual_token_importance_reason], -1)   # 果->因


            # new_input_ids_ = outputs['sequences'][0]
            # new_token_num = outputs['sequences'].shape[-1]
            # new_input_embedding = torch.concatenate((input_embeds, self.language_model.get_input_embeddings()(new_input_ids_).unsqueeze(0)), dim=1)
            # new_attention_mask = torch.concatenate((attention_mask, torch.ones((1, new_input_ids_.shape[0]), device=attention_mask.device, dtype=attention_mask.dtype)), dim=-1)
            # new_input_ids = torch.concatenate((input_ids, new_input_ids_), dim=-1)
            # consistency_generate_kwargs['inputs_embeds'] = new_input_embedding
            # consistency_generate_kwargs['attention_mask'] = new_attention_mask
            # consistency_generate_kwargs['output_scores'] = False
            # consistency_generate_kwargs['output_attentions'] = False
            # consistency_generate_kwargs = self.language_model._get_initial_cache_position(new_input_ids, consistency_generate_kwargs)

            # model_inputs = self.language_model.prepare_inputs_for_generation(new_input_ids,  **consistency_generate_kwargs)
            # consistency_output = self.language_model.forward(**model_inputs, return_dict=True)
            # consistency_score = torch.gather(consistency_output['logits'][:, -new_token_num-1:-1, :].softmax(dim=-1), index=new_input_ids_[None, :, None], dim=-1)

            # consistency_score = torch.prod(consistency_score)


            # return outputs, consistency_score, visual_token_importance


        
        else:
            return self.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_cache=True,
                **generate_kwargs,
            )

    @torch.no_grad()
    def causalTestGenerate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,   # 图像像素点 5,3,448,448
            input_ids: Optional[torch.FloatTensor] = None,  # 1340 待输入token，已嵌入text，未嵌入image
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            large_model: Optional[bool] = False,
            question_ids_test = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values) # 图像特征提取 5,3,448,448->5,256,2048   256应该是16*16的patch，2048是embedding维度即信息量

            image_split_num = vit_embeds.shape[0]
            # average_vit_embeds = torch.mean(vit_embeds, dim=2)  # 对第三维度求平均，测试用，结束后删除  qtyqtyqty
            # normalized_embeds = torch.nn.functional.normalize(average_vit_embeds, p=2, dim=1) # 归一化处理，不然太乱了看不清楚  qtyqtyqty

            input_embeds = self.language_model.get_input_embeddings()(input_ids)    # 1 1340 -> 1 1340 2048     2048应该是embedding维度，即单个token的信息量
            B, N, C = input_embeds.shape    # 1 1340 2048
            input_embeds = input_embeds.reshape(B * N, C)   # 1340 2048
            input_ids = input_ids.reshape(B * N)    # 1340
            selected = (input_ids == self.img_context_token_id) # 将非文本位置设为True，已有文本位置为False
            torch.set_printoptions(threshold=float('inf'))
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)  # 将图片token嵌入进input_input_embeds中 vit_embeds.reshape(-1, C)后大小为1280*2048  -1是 PyTorch 中的一个特殊值，表示自动计算该维度的大小，保持总元素数不变
            input_embeds = input_embeds.reshape(B, N, C)    # 1 1340 2048   input_embeds为最后要的嵌入token
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)   



        if not large_model:
            consistency_generate_kwargs = generate_kwargs.pop('consistency_config')
            generate_kwargs['visual_token_index'] = consistency_generate_kwargs['visual_token_index']
            outputs = self.language_model.generate(     # self.language_model为InternLM2ForCausalLM
                inputs_embeds=input_embeds, # 1340, 2048
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_cache=True,
                **generate_kwargs,
            )
            # 获取聚合结果
            #import pdb; pdb.set_trace()
            visual_token_importance = outputs.aggregated_viusal_token_attention
            consistency_generate_kwargs['visual_token_importance'] = visual_token_importance

            # visual_token_importance = visual_token_importance.unsqueeze(0)
            # _, reasonLen = visual_token_importance.shape
            # start_idx = int(reasonLen * (1 - 1/image_split_num))      # 计算截取初始位置 
            # visual_token_importance = visual_token_importance[:, start_idx:]  # 截取完整的图片的位置

            # 推理
            return write_jsonl_test([visual_token_importance], question_ids_test) # 因->果

            # new_input_ids_ = outputs['sequences'][0]
            # new_token_num = outputs['sequences'].shape[-1]
            # new_input_embedding = torch.concatenate((input_embeds, self.language_model.get_input_embeddings()(new_input_ids_).unsqueeze(0)), dim=1)
            # new_attention_mask = torch.concatenate((attention_mask, torch.ones((1, new_input_ids_.shape[0]), device=attention_mask.device, dtype=attention_mask.dtype)), dim=-1)
            # new_input_ids = torch.concatenate((input_ids, new_input_ids_), dim=-1)
            # consistency_generate_kwargs['inputs_embeds'] = new_input_embedding
            # consistency_generate_kwargs['attention_mask'] = new_attention_mask
            # consistency_generate_kwargs['output_scores'] = False
            # consistency_generate_kwargs['output_attentions'] = False
            # consistency_generate_kwargs = self.language_model._get_initial_cache_position(new_input_ids, consistency_generate_kwargs)

            # model_inputs = self.language_model.prepare_inputs_for_generation(new_input_ids,  **consistency_generate_kwargs)
            # consistency_output = self.language_model.forward(**model_inputs, return_dict=True)
            # consistency_score = torch.gather(consistency_output['logits'][:, -new_token_num-1:-1, :].softmax(dim=-1), index=new_input_ids_[None, :, None], dim=-1)

            # consistency_score = torch.prod(consistency_score)


            # return outputs, consistency_score, visual_token_importance


        
        else:
            return self.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_cache=True,
                **generate_kwargs,
            )

# ============ MODIFY START ============
    # 修改原因：将训练数据写入json文件
    # 修改人：Taoyu Qian
    # 修改时间：2026-02-02
def tensor_to_serializable(obj):
    """
    递归地将 PyTorch tensor / numpy array 转为 Python 原生类型（list / float / int）。
    支持嵌套的 list、tuple、dict 结构。
    """
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu()
        if obj.dim() == 0:
            return obj.item()
        return obj.tolist()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        converted = [tensor_to_serializable(item) for item in obj]
        return converted if isinstance(obj, list) else tuple(converted)

    # int / float / str / None 等原生类型直接返回
    return obj

import numpy as np
import math

def write_jsonl_test(data_list, question_ids_test, initial_ratio=0.43, keep_ratio=0):
    """
    Args:
        data_list: 输入数据列表
        question_ids_test: 测试问题ID列表
        initial_ratio: 第一阶段保留比例，按激活值Top-K直接保留（如0.12=前12%高激活token）
        keep_ratio:    第二阶段保留比例，从剩余token中按因果得分恢复（如0.8=从剩余88%中取前80%因果token）
        最终保留率约为: initial_ratio + (1 - initial_ratio) * keep_ratio
    """
    processed = [tensor_to_serializable(x) for x in data_list]
    processed = processed[0]
    processed[-1] = 0.1
    total_len = len(processed)
    num_images = total_len // 256

    assert num_images >= 1, "至少要有一张图"

    # ============ 单图情况 ============
    if num_images == 1:
        fin_mask_all = _compute_mask_single(
            processed[:256],
            initial_ratio, keep_ratio
        )

        assert len(fin_mask_all) == total_len, \
            f"蒙版长度({len(fin_mask_all)})和token长度({total_len})不匹配"

        #_print_stats("[单图模式]", fin_mask_all, total_len, initial_ratio, keep_ratio)
        return fin_mask_all

    # ============ 多图情况 ============
    num_splits = num_images - 1

    def factor_pairs(n):
        pairs = []
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                pairs.append((i, n // i))
        return pairs

    pairs = factor_pairs(num_splits)
    grid_rows, grid_cols = min(pairs, key=lambda x: abs(x[0] - x[1]))

    print(f"[多图模式] 图片总数: {num_images}, 拼图数量: {num_splits}, 网格布局: {grid_rows} x {grid_cols}")

    # 缩略图因果推理
    last_img_tokens = processed[-256:]
    last_img_tokens[-1] = 0.1

    thumb_mask = _compute_mask_single(
        last_img_tokens,
        initial_ratio, keep_ratio,
        label="[缩略图]"
    )
    thumb_mask = np.array(thumb_mask)
    thumb_mask_2d = thumb_mask.reshape(16, 16)

    sub_h = math.ceil(16 / grid_rows)
    sub_w = math.ceil(16 / grid_cols)

    fin_mask_all = []

    for block_idx in range(num_splits):
        r = block_idx // grid_cols
        c = block_idx % grid_cols

        start_row = r * sub_h
        end_row   = min(start_row + sub_h, 16)
        start_col = c * sub_w
        end_col   = min(start_col + sub_w, 16)

        sub_region = thumb_mask_2d[start_row:end_row, start_col:end_col]

        # block_mask = np.repeat(sub_region, repeats=math.ceil(16 / sub_region.shape[0]), axis=0)
        # block_mask = np.repeat(block_mask, repeats=math.ceil(16 / sub_region.shape[1]), axis=1)
        # block_mask = block_mask[:16, :16].astype(int)
        # 防止 sub_region 为空导致除零
        if sub_region.shape[0] == 0 or sub_region.shape[1] == 0:
            block_mask = np.ones((16, 16), dtype=int)
        else:
            block_mask = np.repeat(sub_region, repeats=math.ceil(16 / sub_region.shape[0]), axis=0)
            block_mask = np.repeat(block_mask, repeats=math.ceil(16 / sub_region.shape[1]), axis=1)
            block_mask = block_mask[:16, :16].astype(int)

        assert block_mask.size == 256, f"切片mask大小不对: {block_mask.shape}"
        fin_mask_all.extend(block_mask.flatten().tolist())

    fin_mask_all.extend(thumb_mask.tolist())

    assert len(fin_mask_all) == total_len, \
        f"蒙版长度({len(fin_mask_all)})和token长度({total_len})不匹配"

    #_print_stats("[多图模式汇总]", fin_mask_all, total_len, initial_ratio, keep_ratio)
    return fin_mask_all

def _compute_mask_single(img_tokens_raw, initial_ratio, keep_ratio, label=None):
    """
    核心双阶段mask计算：
      Stage-1: 按激活值 Top initial_ratio 直接保留 → initial_set
      Stage-2: 从剩余token输入CausalInference，按因果得分恢复 Top keep_ratio → causal_set
      最终mask = initial_set ∪ causal_set
    """
    img_tokens = list(img_tokens_raw)
    img_tokens[-1] = 0.1
    n = len(img_tokens)  # 应为256

    # ── Stage 1: 按激活值直接保留 Top initial_ratio ──────────────────────
    n_keep_init = max(1, int(n * initial_ratio))
    threshold_init = np.partition(img_tokens, -n_keep_init)[-n_keep_init]
    
    # initial_set: 第一阶段直接保留的token索引
    initial_set = set(
        i for i in range(n) if img_tokens[i] >= threshold_init
    )

    # ── Stage 2: 对剩余token做因果推理，恢复因果相关token ────────────────
    # 构造因果模型输入：仅使用第一阶段保留的激活值（置0其余）
    # 注意：这里_chunk只是CausalInference的输入信号，不代表保留决策
    _chunk = [img_tokens[i] if i in initial_set else 0.0 for i in range(n)]

    anchor_sizes = [(4, 4), (3, 5), (5, 3)]
    anchors = generate_custom_anchors(img_tokens, anchor_sizes)

    num_anchors = len(anchors)
    batch_x = np.array([_chunk] * num_anchors, dtype=np.float32)
    batch_y = np.array(anchors, dtype=np.float32)

    result = CausalInferencePredictor(
        batch_x, batch_y,
        'causalUtils/models/CaVIN_3.22M.pth',
        threshold='0.1',
    )

    token_mask = anchors_to_token_mask(img_tokens, anchor_sizes, result['binary_mask'])

    # 因果得分：第一阶段已保留的token不参与第二阶段竞争（避免重复计数）
    scores = np.array([abs(token_mask[i]) for i in range(n)], dtype=np.float32)
    
    # 第一阶段已保留的token得分清零，仅从"剩余token"中按因果恢复
    for i in initial_set:
        scores[i] = -1.0  # 排除出候选池

    remaining_indices = [i for i in range(n) if i not in initial_set]
    n_remaining = len(remaining_indices)
    
    # 从剩余token中按因果得分恢复 Top keep_ratio
    n_keep_causal = max(0, int(n_remaining * keep_ratio))

    if n_keep_causal > 0:
        sorted_indices = np.argsort(-scores)  # 降序
        # 只从remaining_indices中取，initial_set的得分为-1会排在后面
        causal_set = set(sorted_indices[:n_keep_causal].tolist())
        # 确保不包含initial_set（防御性）
        causal_set -= initial_set
    else:
        causal_set = set()

    # ── 合并：并集 ────────────────────────────────────────────────────────
    final_set = initial_set | causal_set

    fin_mask = [1 if i in final_set else 0 for i in range(n)]

    # 打印统计
    if label:
        fin_token = sum(fin_mask)
        expected_ratio = initial_ratio + (1 - initial_ratio) * keep_ratio
        # print(f"{label}")
        # print(f"  Stage-1 (initial_ratio={initial_ratio:.2f}): {n}个token → 直接保留 {len(initial_set)} 个高激活token")
        # print(f"  Stage-2 (keep_ratio={keep_ratio:.2f}):      剩余{n_remaining}个token → 因果恢复 {len(causal_set)} 个token")
        # print(f"  理论最终保留率: {initial_ratio:.2f} + (1-{initial_ratio:.2f})×{keep_ratio:.2f} = {expected_ratio:.4f}")
        # print(f"  实际最终保留率: {fin_token}/{n} = {fin_token/n:.4f}")

    return fin_mask

def _print_stats(label, fin_mask_all, total_len, initial_ratio, keep_ratio):
    fin_token = sum(fin_mask_all)
    actual_ratio = fin_token / total_len
    expected_ratio = initial_ratio + (1 - initial_ratio) * keep_ratio
    print(f"{label}")
    print(f"  理论最终保留率: {initial_ratio:.2f} + (1-{initial_ratio:.2f})×{keep_ratio:.2f} = {expected_ratio:.4f}")
    print(f"  实际最终保留率: {fin_token}/{total_len} = {actual_ratio:.4f}")
    print(f"  剪枝率: {100*(total_len-fin_token)/total_len:.2f}% ({total_len-fin_token}/{total_len} 被剪掉)")