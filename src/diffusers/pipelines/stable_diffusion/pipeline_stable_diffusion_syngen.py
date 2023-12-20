# Copyright 2023 The SynGen Authors and The HuggingFace Team. All rights reserved.
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

from typing import Any, Callable, Dict, Optional, Union, List, Tuple

import torch
import torch.distributions as dist
import itertools
import numpy as np
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import KarrasDiffusionSchedulers
from ...models.attention_processor import Attention
from ...utils import logging, replace_example_docstring
from . import (
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
    StableDiffusionPipeline,
)
from .pipeline_stable_diffusion import EXAMPLE_DOC_STRING, rescale_noise_cfg


logger = logging.get_logger(__name__)


class StableDiffusionSynGenPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )

        self.parser = None

    def _aggregate_and_get_attention_maps_per_token(self):
        attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"),
        )
        attention_maps_list = _get_attention_maps_list(attention_maps=attention_maps)
        return attention_maps_list

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> (
        torch.Tensor
    ):
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - step_size * grad_cond
        return latents

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteAttnProcessor(
                attnstore=self.attention_store, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    # Based on StableDiffusionPipeline.__call__ . New code is annotated with NEW.
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        attn_res=None,
        subtrees_indices: Optional[List[List[str]]] = None,
        syngen_step_size: float = 20.0,
        num_intervention_steps: int = 25,
        include_entities: bool = False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            attn_res (`tuple`, *optional*, default computed from width and height):
                The 2D resolution of the semantic attention map.
            subtrees_indicies (`List[List[str]]`, *optional*, default computed from spacy parser):
                Index pairs mapping between entities and their visual attributes in the prompt.
            syngen_step_size (`float`, *optional*, default to 20.0):
                Controls the step size of each SynGen update.
            num_intervention_steps ('int', *optional*, defaults to 25):
                The number of times we apply SynGen.
            include_entities ('bool', *optional*, defaults to `False`):
                Whether or not to include entities in the loss calculation.


        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # NEW - stores the attention calculated in the unet
        if attn_res is None:
            attn_res = int(np.ceil(width / 32)), int(np.ceil(height / 32))
        self.attention_store = AttentionStore(attn_res)
        self.register_attention_control()

        text_embeddings = (
            prompt_embeds[batch_size * num_images_per_prompt :]
            if do_classifier_free_guidance
            else prompt_embeds
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # NEW
                if i < num_intervention_steps:
                    latents = self._syngen_step(
                        latents,
                        text_embeddings,
                        t,
                        i,
                        syngen_step_size,
                        cross_attention_kwargs,
                        prompt,
                        subtrees_indices,
                        include_entities=include_entities,
                        num_intervention_steps=num_intervention_steps,
                    )

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if do_classifier_free_guidance
                        else latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=guidance_rescale,
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    def _syngen_step(
        self,
        latents,
        text_embeddings,
        t,
        i,
        step_size,
        cross_attention_kwargs,
        prompt,
        subtrees_indices,
        include_entities: bool,
        num_intervention_steps: int,
    ):
        with torch.enable_grad():
            latents = latents.clone().detach().requires_grad_(True)
            updated_latents = []
            for latent, text_embedding in zip(latents, text_embeddings):
                # Forward pass of denoising with text conditioning
                latent = latent.unsqueeze(0)
                text_embedding = text_embedding.unsqueeze(0)

                self.unet(
                    latent,
                    t,
                    encoder_hidden_states=text_embedding,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                self.unet.zero_grad()
                # Get attention maps
                attention_maps = self._aggregate_and_get_attention_maps_per_token()
                loss = self._compute_loss(
                    attention_maps=attention_maps,
                    prompt=prompt,
                    subtrees_indices=subtrees_indices,
                    include_entities=include_entities,
                )
                # Perform gradient update
                if i < num_intervention_steps:
                    if loss != 0:
                        latent = self._update_latent(
                            latents=latent, loss=loss, step_size=step_size
                        )
                    logger.info(f"NOT_YET_IMPLEMENTED_ExprJoinedStr")

            updated_latents.append(latent)

        latents = torch.cat(updated_latents, dim=0)

        return latents

    def _compute_loss(
        self,
        attention_maps: List[torch.Tensor],
        prompt: Union[str, List[str]],
        subtrees_indices,
        include_entities: bool,
    ) -> torch.Tensor:
        attn_map_idx_to_wp = get_attention_map_index_to_wordpiece(
            self.tokenizer, prompt
        )
        loss = self._attribution_loss(
            attention_maps,
            prompt,
            attn_map_idx_to_wp,
            subtrees_indices,
            include_entities,
        )

        return loss

    def _attribution_loss(
        self,
        attention_maps: List[torch.Tensor],
        prompt: Union[str, List[str]],
        attn_map_idx_to_wp,
        subtrees_indices,
        include_entities: bool,
    ) -> torch.Tensor:
        if not subtrees_indices:
            self._load_parser()

            subtrees_indices = self._extract_attribution_indices(
                prompt, include_entities
            )
        loss = 0

        for subtree_indices in subtrees_indices:
            noun, modifier = split_indices(subtree_indices)
            all_subtree_pairs = list(itertools.product(noun, modifier))
            if noun and not modifier:
                if isinstance(noun, list) and len(noun) == 1:
                    processed_noun = noun[0]
                else:
                    processed_noun = noun
                loss += calculate_negative_loss(
                    attention_maps,
                    modifier,
                    processed_noun,
                    subtree_indices,
                    attn_map_idx_to_wp,
                )
            else:
                positive_loss, negative_loss = self._calculate_losses(
                    attention_maps,
                    all_subtree_pairs,
                    subtree_indices,
                    attn_map_idx_to_wp,
                )

                loss += positive_loss
                loss += negative_loss

        return loss

    def _load_parser(self):
        if self.parser is None:
            try:
                import spacy
                self.parser = spacy.load("en_core_web_trf")
            except ImportError as e:
                raise ImportError(
                    "Running this pipeline requires the `spacy` package to be installed. Run `pip install spacy`."
                ) from e

    def _calculate_losses(
        self,
        attention_maps,
        all_subtree_pairs,
        subtree_indices,
        attn_map_idx_to_wp,
    ):
        positive_loss = []
        negative_loss = []
        for pair in all_subtree_pairs:
            noun, modifier = pair
            positive_loss.append(
                calculate_positive_loss(attention_maps, modifier, noun)
            )
            negative_loss.append(
                calculate_negative_loss(
                    attention_maps, modifier, noun, subtree_indices, attn_map_idx_to_wp
                )
            )

        positive_loss = sum(positive_loss)
        negative_loss = sum(negative_loss)

        return positive_loss, negative_loss

    def _align_indices(self, prompt, spacy_pairs):
        wordpieces2indices = get_indices(self.tokenizer, prompt)
        paired_indices = []
        collected_spacy_indices = set()  # helps track recurring nouns across different relations (i.e., cases where there is more than one instance of the same word)

        for pair in spacy_pairs:
            curr_collected_wp_indices = []  # helps track which nouns and amods were added to the current pair (this is useful in sentences with repeating amod on the same relation (e.g., "a red red red bear"))
            for member in pair:
                for idx, wp in wordpieces2indices.items():
                    if wp in [start_token, end_token]:
                        continue

                    wp = wp.replace(wordpiece_token, "")
                    if member.text.lower() == wp.lower():
                        if (
                            idx not in curr_collected_wp_indices
                            and idx not in collected_spacy_indices
                        ):
                            curr_collected_wp_indices.append(idx)
                            break
                    # take care of wordpieces that are split up
                    elif member.text.lower().startswith(
                        wp.lower()
                    ) and wp.lower() != member.text.lower():  # can maybe be while loop
                        wp_indices = align_wordpieces_indices(
                            wordpieces2indices, idx, member.text
                        )
                        # check if all wp_indices are not already in collected_spacy_indices
                        if (
                            wp_indices
                            and (wp_indices not in curr_collected_wp_indices)
                            and all(
                                wp_idx not in collected_spacy_indices
                                for wp_idx in wp_indices
                            )
                        ):
                            curr_collected_wp_indices.append(wp_indices)
                            break

            for collected_idx in curr_collected_wp_indices:
                if isinstance(collected_idx, list):
                    for idx in collected_idx:
                        collected_spacy_indices.add(idx)
                else:
                    collected_spacy_indices.add(collected_idx)

            if curr_collected_wp_indices:
                paired_indices.append(curr_collected_wp_indices)
            else:
                print(f"NOT_YET_IMPLEMENTED_ExprJoinedStr")

        return paired_indices

    def _extract_attribution_indices(self, prompt: str, include_entities: bool):
        """
        Find syntactic <entity, visual-attribute> pairs in the prompt
        """

        doc = self.parser(prompt)

        modifier_indices = []
        # extract standard attribution indices
        modifier_sets_1 = extract_attribution_indices(doc)
        modifier_indices_1 = self._align_indices(prompt, modifier_sets_1)
        if modifier_indices_1:
            modifier_indices.append(modifier_indices_1)

        # extract attribution indices with verbs in between
        modifier_sets_2 = extract_attribution_indices_with_verb_root(doc)
        modifier_indices_2 = self._align_indices(prompt, modifier_sets_2)
        if modifier_indices_2:
            modifier_indices.append(modifier_indices_2)

        modifier_sets_3 = extract_attribution_indices_with_verbs(doc)
        modifier_indices_3 = self._align_indices(prompt, modifier_sets_3)
        if modifier_indices_3:
            modifier_indices.append(modifier_indices_3)

        # entities only
        if include_entities:
            modifier_sets_4 = extract_entities_only(doc)
            modifier_indices_4 = self._align_indices(prompt, modifier_sets_4)
            modifier_indices.append(modifier_indices_4)

        # make sure there are no duplicates
        modifier_indices = unify_lists(modifier_indices)
        print(f"NOT_YET_IMPLEMENTED_ExprJoinedStr")

        return modifier_indices


def _get_attention_maps_list(attention_maps: torch.Tensor) -> List[torch.Tensor]:
    attention_maps *= 100
    attention_maps_list = [
        attention_maps[:, :, i] for i in range(attention_maps.shape[2])
    ]

    return attention_maps_list


def unify_lists(list_of_lists):
    def flatten(lst):
        for elem in lst:
            if isinstance(elem, list):
                yield from flatten(elem)
            else:
                yield elem

    def have_common_element(lst1, lst2):
        flat_list1 = set(flatten(lst1))
        flat_list2 = set(flatten(lst2))
        return not flat_list1.isdisjoint(flat_list2)

    lst = []
    for l in list_of_lists:
        lst += l
    changed = True
    while changed:
        changed = False
        merged_list = []
        while lst:
            first = lst.pop(0)
            was_merged = False
            for index, other in enumerate(lst):
                if have_common_element(first, other):
                    # If we merge, we should flatten the other list but not first
                    new_merged = first + [item for item in other if item not in first]
                    lst[index] = new_merged
                    changed = True
                    was_merged = True
                    break
            if not was_merged:
                merged_list.append(first)
        lst = merged_list

    return lst


start_token = "<|startoftext|>"
end_token = "<|endoftext|>"
wordpiece_token = "</w>"


def _get_outside_indices(subtree_indices, attn_map_idx_to_wp):
    flattened_subtree_indices = _flatten_indices(subtree_indices)
    outside_indices = [
        map_idx
        for map_idx in attn_map_idx_to_wp.keys()
        if (map_idx not in flattened_subtree_indices)
    ]
    return outside_indices


def _flatten_indices(related_indices):
    flattened_related_indices = []
    for item in related_indices:
        if isinstance(item, list):
            flattened_related_indices.extend(item)
        else:
            flattened_related_indices.append(item)
    return flattened_related_indices


def split_indices(related_indices: List[int]) -> Tuple[str, str]:
    noun = [related_indices[-1]]  # assumes noun is always last in the list
    modifier = related_indices[:-1]
    if isinstance(modifier, int):
        modifier = [modifier]
    return noun, modifier


def _symmetric_kl(attention_map1, attention_map2):
    # Convert map into a single distribution: 16x16 -> 256
    if len(attention_map1.shape) > 1:
        attention_map1 = attention_map1.reshape(-1)
    if len(attention_map2.shape) > 1:
        attention_map2 = attention_map2.reshape(-1)

    p = dist.Categorical(probs=attention_map1)
    q = dist.Categorical(probs=attention_map2)

    kl_divergence_pq = dist.kl_divergence(p, q)
    kl_divergence_qp = dist.kl_divergence(q, p)

    avg_kl_divergence = (kl_divergence_pq + kl_divergence_qp) / 2
    return avg_kl_divergence


def calculate_positive_loss(attention_maps, modifier, noun):
    src_indices = modifier
    dest_indices = noun

    if isinstance(src_indices, list) and isinstance(dest_indices, list):
        wp_pos_loss = [
            _symmetric_kl(attention_maps[s], attention_maps[d])
            for (s, d) in itertools.product(src_indices, dest_indices)
        ]
        positive_loss = max(wp_pos_loss)
    elif isinstance(dest_indices, list):
        wp_pos_loss = [
            _symmetric_kl(attention_maps[src_indices], attention_maps[d])
            for d in dest_indices
        ]
        positive_loss = max(wp_pos_loss)
    elif isinstance(src_indices, list):
        wp_pos_loss = [
            _symmetric_kl(attention_maps[s], attention_maps[dest_indices])
            for s in src_indices
        ]
        positive_loss = max(wp_pos_loss)
    else:
        positive_loss = _symmetric_kl(
            attention_maps[src_indices], attention_maps[dest_indices]
        )

    return positive_loss


def _calculate_outside_loss(attention_maps, src_indices, outside_loss):
    negative_loss = []
    computed_pairs = set()
    pair_counter = 0

    for outside_idx in outside_loss:
        if isinstance(src_indices, list):
            wp_neg_loss = []
            for t in src_indices:
                pair_key = (t, outside_idx)
                if pair_key not in computed_pairs:
                    wp_neg_loss.append(
                        _symmetric_kl(attention_maps[t], attention_maps[outside_idx])
                    )
                    computed_pairs.add(pair_key)
            negative_loss.append(max(wp_neg_loss) if wp_neg_loss else 0)
            pair_counter += 1

        else:
            pair_key = (src_indices, outside_idx)
            if pair_key not in computed_pairs:
                negative_loss.append(
                    _symmetric_kl(
                        attention_maps[src_indices], attention_maps[outside_idx]
                    )
                )
                computed_pairs.add(pair_key)
                pair_counter += 1

    return negative_loss, pair_counter


def align_wordpieces_indices(wordpieces2indices, start_idx, target_word):
    """
    Aligns a `target_word` that contains more than one wordpiece (the first wordpiece is `start_idx`)
    """

    wp_indices = [start_idx]
    wp = wordpieces2indices[start_idx].replace(wordpiece_token, "")

    # Run over the next wordpieces in the sequence (which is why we use +1)
    for wp_idx in range(start_idx + 1, len(wordpieces2indices)):
        if wp.lower() == target_word.lower():
            break

        wp2 = wordpieces2indices[wp_idx].replace(wordpiece_token, "")
        if target_word.lower().startswith(
            wp.lower() + wp2.lower()
        ) and wp2.lower() != target_word.lower():
            wp += wordpieces2indices[wp_idx].replace(wordpiece_token, "")
            wp_indices.append(wp_idx)
        else:
            wp_indices = []  # if there's no match, you want to clear the list and finish
            break

    return wp_indices


def extract_attribution_indices(doc):
    """
    Find syntactic <entity, visual-attribute> pairs in the prompt. For example, "a red dog".
    """

    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp"]

    for w in doc:
        if w.pos_ not in ["NOUN", "PROPN"] or w.dep_ in modifiers:
            continue
        subtree = []
        stack = []
        for child in w.children:
            if child.dep_ in modifiers:
                subtree.append(child)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(w)
            subtrees.append(subtree)
    return subtrees


def extract_attribution_indices_with_verbs(doc):
    """
    This function specifically addresses cases where a verb is between
       a noun and its modifier. For instance: "a dog that is red"
       here, the aux is between 'dog' and 'red'.
    """

    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp", "relcl"]
    for w in doc:
        if w.pos_ not in ["NOUN", "PROPN"] or w.dep_ in modifiers:
            continue
        subtree = []
        stack = []
        for child in w.children:
            if child.dep_ in modifiers:
                if child.pos_ not in ["AUX", "VERB"]:
                    subtree.append(child)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                # we don't want to add 'is' or other verbs to the loss, we want their children
                if node.pos_ not in ["AUX", "VERB"]:
                    subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(w)
            subtrees.append(subtree)
        return subtrees


def extract_attribution_indices_with_verb_root(doc):
    """
    This function specifically addresses cases where a verb is between
       a noun and its modifier. For instance: "a dog that is red"
       here, the aux is between 'dog' and 'red'.
    """

    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp"]

    for w in doc:
        subtree = []
        stack = []

        # if w is a verb/aux and has a noun child and a modifier child, add them to the stack
        if w.pos_ != "AUX" or w.dep_ in modifiers:
            continue

        for child in w.children:
            if child.dep_ in modifiers or child.pos_ in ["NOUN", "PROPN"]:
                if child.pos_ not in ["AUX", "VERB"]:
                    subtree.append(child)
                stack.extend(child.children)
        # did not find a pair of noun and modifier
        if len(subtree) < 2:
            continue

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                # we don't want to add 'is' or other verbs to the loss, we want their children
                if node.pos_ not in ["AUX"]:
                    subtree.append(node)
                stack.extend(node.children)

        if subtree:
            if w.pos_ not in ["AUX"]:
                subtree.append(w)
            subtrees.append(subtree)
    return subtrees


def extract_entities_only(doc):
    entities = []
    for w in doc:
        if w.pos_ in ["NOUN", "PROPN"]:
            entities.append([w])
    return entities


def calculate_negative_loss(
    attention_maps, modifier, noun, subtree_indices, attn_map_idx_to_wp
):
    outside_indices = _get_outside_indices(subtree_indices, attn_map_idx_to_wp)

    negative_noun_loss, num_noun_pairs = _calculate_outside_loss(
        attention_maps, noun, outside_indices
    )
    if outside_indices:
        negative_noun_loss = -sum(negative_noun_loss) / len(outside_indices)
    else:
        negative_noun_loss = 0

    if modifier:
        negative_modifier_loss, num_modifier_pairs = _calculate_outside_loss(
            attention_maps, modifier, outside_indices
        )
        if outside_indices:
            negative_modifier_loss = -sum(negative_modifier_loss) / len(outside_indices)
        else:
            negative_modifier_loss = 0

        negative_loss = (negative_modifier_loss + negative_noun_loss) / 2
    else:
        negative_loss = negative_noun_loss

    return negative_loss


def get_indices(tokenizer, prompt: str) -> Dict[str, int]:
    """Utility function to list the indices of the tokens you wish to alter"""
    ids = tokenizer(prompt).input_ids
    indices = {
        i: tok for tok, i in zip(tokenizer.convert_ids_to_tokens(ids), range(len(ids)))
    }
    return indices


def get_attention_map_index_to_wordpiece(tokenizer, prompt):
    attn_map_idx_to_wp = {}

    wordpieces2indices = get_indices(tokenizer, prompt)

    # Ignore `start_token` and `end_token`
    for i in list(wordpieces2indices.keys())[1:-1]:
        wordpiece = wordpieces2indices[i]
        wordpiece = wordpiece.replace(wordpiece_token, "")
        attn_map_idx_to_wp[i] = wordpiece

    return attn_map_idx_to_wp


class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {"down": [], "mid": [], "up": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0 and is_cross:
            if attn.shape[1] == np.prod(self.attn_res):
                self.step_store[place_in_unet].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def aggregate_attention(self, from_where: List[str]) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item in attention_maps[location]:
                cross_maps = item.reshape(
                    -1, self.attn_res[0], self.attn_res[1], item.shape[-1]
                )
                out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res


class AttendExciteAttnProcessor:
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # only need to store attention maps during the Attend and Excite process
        if attention_probs.requires_grad:
            self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
