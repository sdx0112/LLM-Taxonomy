import requests
from typing import Dict, List
import json
from langchain.llms.base import LLM, Optional, List, Mapping, Any
from src.constants import *

class Llama2Embeddings:
    def get_embedding(self, x):
        data = {'query': x, 'task': 'Embedding'}
        # r = requests.post(url=llama2_url, json=json.dumps(data))
        with requests.post(url=llama2_url, json=json.dumps(data), stream=True) as r:
            response = r.json()['result']
        return response

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # print('Embed documents')
        return [self.get_embedding(x) for x in texts]

    def embed_query(self, text):
        return self.get_embedding(text)


class Llama2LLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]]=None) -> str:
        data = {
            "prompt": prompt,
            "use_story": False,
            "use_authors_note": False,
            "use_world_info": False,
            "use_memory": False,
            "max_context_length": 4000,
            "max_length": 512,
            "rep_pen": 1.12,
            "rep_pen_range": 1024,
            "rep_pen_slope": 0.9,
            "temperature": 0.6,
            "tfs": 0.9,
            "top_p": 0.95,
            "top_k": 0.6,
            "typical": 1,
            "frmttriminc": True
        }

        # Add the stop sequences to the data if they are provided
        if stop is not None:
            data["stop_sequence"] = stop

        # Send a POST request to the Kobold API with the data
        data = {'query': prompt, 'task': 'Completion'}
        # r = requests.post(url=llama2_url, json=json.dumps(data))
        with requests.post(url=llama2_url, json=json.dumps(data), stream=True) as r:
            response = r.json()['result']
        return response # r.json()['result']


    def __call__(self, prompt: str, stop: Optional[List[str]]=None) -> str:
        return self._call(prompt, stop)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}