<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# SynGen

SynGen for Stable Diffusion was proposed in [Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment](https://royira.github.io/SynGen/) and mitigates incorrect associations between entities and their visual attributes.

The abstract from the paper is:

*Text-conditioned image generation models often generate incorrect associations between entities and their visual attributes. This reflects an impaired mapping between linguistic binding of entities and modifiers in the prompt and visual binding of the corresponding elements in the generated image. As one notable example, the query "a pink sunflower and a yellow flamingo" may incorrectly produce an image of a pink flamingo and a yellow sunflower. To remedy this issue, we propose SynGen, an approach which first syntactically analyses the prompt to identify entities and their modifiers, and then uses a novel loss function that encourages the cross-attention maps to agree with the linguistic binding reflected by the syntax. Specifically, we encourage large overlap between attention maps of entities and their modifiers, and small overlap with other entities and modifier words. The loss is optimized during inference, without retraining the model. Human evaluation on three datasets, including one new and challenging set, demonstrate significant improvements of SynGen compared with current state of the art methods. This work highlights how making use of sentence structure during inference can efficiently and substantially improve the faithfulness of text-to-image generation.*

You can find additional information about SynGen on the [project page](https://royira.github.io/SynGen/), the [original codebase](https://github.com/RoyiRa/Syntax-Guided-Generation), or try it out in a [demo](https://huggingface.co/spaces/Royir/SynGen).

<Tip>

SynGen is based on the dependency parsing of [spaCy](https://spacy.io/usage), which is not downloaded with `diffusers`. To use this pipeline, you have to run the following commands:
1. `pip install spacy`
2. `python -m spacy download en_core_web_trf`

</Tip>

## StableDiffusionSynGenPipeline

[[autodoc]] StableDiffusionSynGenPipeline
	- all
	- __call__

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput