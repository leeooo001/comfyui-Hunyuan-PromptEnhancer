import os, sys
import re
import time
import logging

import omegaconf
script_directory = os.path.dirname(os.path.abspath(__file__))
CKPTS_FILE = omegaconf.OmegaConf.load(os.path.join(script_directory, 'ckpts.yaml'))


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt_deault = "你是一位图像生成提示词撰写专家，请根据用户输入的提示词，改写生成新的提示词，改写后的提示词要求：1 改写后提示词包含的主体/动作/数量/风格/布局/关系/属性/文字等 必须和改写前的意图一致； 2 在宏观上遵循“总-分-总”的结构，确保信息的层次清晰；3 客观中立，避免主观臆断和情感评价；4 由主到次，始终先描述最重要的元素，再描述次要和背景元素；5 逻辑清晰，严格遵循空间逻辑或主次逻辑，使读者能在大脑中重建画面；6 结尾点题，必须用一句话总结图像的整体风格或类型。",

def replace_single_quotes(text):
    """
    Replace single quotes within words with double quotes, and convert
    curly single quotes to curly double quotes for consistency.
    """
    pattern = r"\B'([^']*)'\B"
    replaced_text = re.sub(pattern, r'"\1"', text)
    replaced_text = replaced_text.replace("’", "”")
    replaced_text = replaced_text.replace("‘", "“")
    return replaced_text


class HunyuanPromptEnhancer:
    def __init__(self, models_root_path, device_map="auto"):
        if not logging.getLogger(__name__).handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            models_root_path, device_map=device_map, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            models_root_path, trust_remote_code=True
        )

    @torch.inference_mode()
    def predict(
        self,
        prompt_cot,
        sys_prompt,
        temperature=0,
        top_p=1.0,
        max_new_tokens=512,
    ):
        org_prompt_cot = prompt_cot
        try:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": org_prompt_cot},
            ]
            tokenized_chat = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,  # Toggle thinking mode (default: True)
            )
            inputs = tokenized_chat.to(self.model.device)
            do_sample = temperature is not None and float(temperature) > 0
            outputs = self.model.generate(
                inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=do_sample,
                temperature=float(temperature) if do_sample else None,
                top_p=float(top_p) if do_sample else None,
            )

            generated_sequence = outputs[0]
            prompt_length = inputs.shape[-1]
            new_tokens = generated_sequence[prompt_length:]
            output_res = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            answer_pattern = r"<answer>(.*?)</answer>"
            answer_matches = re.findall(answer_pattern, output_res, re.DOTALL)
            if answer_matches:
                prompt_cot = answer_matches[0].strip()
            else:
                output_clean = re.sub(r"<think>[\s\S]*?</think>", "", output_res)
                output_clean = output_clean.strip()
                prompt_cot = output_clean if output_clean else org_prompt_cot
            prompt_cot = replace_single_quotes(prompt_cot)
            self.logger.info("Re-prompting succeeded; using the new prompt")
        except Exception as e:
            prompt_cot = org_prompt_cot
            self.logger.exception("Re-prompting failed; using the original prompt")

        return prompt_cot


class XX_Hunyuan_PromptEnhancer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "models": (["int8","fp16"],),  
                "temperature": ("FLOAT", {"default": 0.7, "step": 0.01, "min": 0, "max": 100.00}),
                "top_p": ("FLOAT", {"default": 0.9, "step": 0.01, "min": 0, "max": 100.00}),
                "max_new_tokens": ("INT", {"default": 256, "step": 1, "min": 0, "max": 99999}),
                "prompt": ("STRING", {"multiline": True, "placeholder": "Prompt Text", "default": prompt_deault}),
                "text": ("STRING", {"multiline": True, "placeholder": "Prompt Text"}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "run"

    CATEGORY="XX"

    def run(self, models, temperature, top_p, max_new_tokens, prompt, text):
        if models == 'fp16':
            CKPTS_PATH = CKPTS_FILE['modelpath_fp16']
        elif models == 'int8':
            CKPTS_PATH = CKPTS_FILE['modelpath_int8']
        else:
            return None

        enhancer = HunyuanPromptEnhancer(models_root_path=CKPTS_PATH, device_map="auto")

        new_prompt = enhancer.predict(
            prompt_cot=text,
            sys_prompt = prompt,
            temperature=0.7,   # >0 enables sampling; 0 uses deterministic generation
            top_p=0.9,
            max_new_tokens=256,
        )
        return(str(new_prompt),)


NODE_CLASS_MAPPINGS = {
    "XX_Hunyuan_PromptEnhancer": XX_Hunyuan_PromptEnhancer,
    }
    

NODE_DISPLAY_NAME_MAPPINGS = {
    "XX_Hunyuan_PromptEnhancer": "XX_Hunyuan_PromptEnhancer",
    }
