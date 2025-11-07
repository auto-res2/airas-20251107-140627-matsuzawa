
LLM Name: o3-2025-04-16
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
license: apache-2.0
language:
- pt
base_model: meta-llama/Llama-2-7b
pipeline_tag: text-generation
---

# MED-LLM-BR: Medical Large Language Models for Brazilian Portuguese
MED-LLM-BR is a collaborative project between [HAILab](https://github.com/HAILab-PUCPR) and [Comsentimento](https://www.comsentimento.com.br/), which aims to develop multiple medical LLMs for Portuguese language, including base models and task-specific models, with different sizes. 

## Introduction
Clinical-BR-LlaMA-2-7B is a fine-tuned language model specifically designed for generating clinical notes in Portuguese. This model builds on the strengths of LlaMA 2 7B, adapting it through targeted fine-tuning techniques to meet the unique demands of clinical text generation. By focusing on the nuances and complexities of medical language in Portuguese, Clinical-BR-LlaMA-2-7B aims to support healthcare professionals with contextually accurate and relevant clinical documentation.

## Fine-Tuning Approach
To enhance memory efficiency and reduce computational demands, we implemented LoRA with 16-bit precision on the q_proj and v_proj projections. We configured LoRA with R set to 8, Alpha to 16, and Dropout to 0.1, allowing the model to adapt effectively while preserving output quality. For optimization, the AdamW optimizer was used with parameters β1 = 0.9 and β2 = 0.999, achieving a balance between fast convergence and training stability. This careful tuning process ensures robust performance in generating accurate and contextually appropriate clinical text in Portuguese.

## Data
The fine-tuning of Clinical-BR-LlaMA-2-7B utilized 2.4GB of text from three clinical datasets. The SemClinBr project provided diverse clinical narratives from Brazilian hospitals, while the BRATECA dataset contributed admission notes from various departments in 10 hospitals. Additionally, data from Lopes et al., 2019, added neurology-focused texts from European Portuguese medical journals. These datasets collectively improved the model’s ability to generate accurate clinical notes in Portuguese.

## Provisional Citation:
```
@inproceedings{pinto2024clinicalLLMs,
  title        = {Developing Resource-Efficient Clinical LLMs for Brazilian Portuguese},
  author       = {João Gabriel de Souza Pinto and Andrey Rodrigues de Freitas and Anderson Carlos Gomes Martins and Caroline Midori Rozza Sawazaki and Caroline Vidal and Lucas Emanuel Silva e Oliveira},
  booktitle    = {Proceedings of the 34th Brazilian Conference on Intelligent Systems (BRACIS)},
  year         = {2024},
  note         = {In press},
}
```
Output:
{
    "extracted_code": ""
}
