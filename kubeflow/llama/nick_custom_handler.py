import os
import json
from typing import List, Dict
from generation import Llama, Dialog
from ts.torch_handler.base_handler import BaseHandler

"""
author : @nick.bae
"""

class LlamaHandler(BaseHandler):

    def __init__(self):
        self.initialized = False
        self.model = None
        self.device = None
        self.ckpt_dir = None
        self.tokenizer_path =  None
        self.max_seq_len = 512
        self.max_batch_size = 8

    def initialize(self, context):

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # load model, tokenizer
        self.ckpt_dir = os.path.join(model_dir, 'model')
        self.tokenizer_path = os.path.join(model_dir, 'tokenizer')
        self.max_seq_len = 512
        self.max_batch_size = 8
        self.generator = Llama.build(
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_path,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
        )
        self.initialized = True

    def preprocess(self, data):
        
        """
        클라이언트로부터 받은 입력 데이터 처리
        """
        
        body = data[0]['body']

        if isinstance(body, str):
            body = json.loads(body)
        
        dialog = body.get('dialog')
        self.temperature = body.get('temperature', 0.6)
        self.top_p = body.get('top_p', 0.9)
        self.max_gen_len = body.get('max_gen_len')
        
        if not isinstance(dialog, list):
            raise ValueError('dialog should be a list')
            
        return dialog

    def inference(self, dialog: List[Dialog]):
        
        """
        모델을 실행하여 대화 생성
        """
        results = self.generator.chat_completion(
            dialog,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return results

    def postprocess(self, inference_output):
        """
        생성된 대화를 클라이언트에게 반환
        """
        return inference_output

_service = LlamaHandler()