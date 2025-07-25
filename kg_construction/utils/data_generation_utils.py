from datetime import datetime
from typing import Dict, Any, Type, List

from pydantic import BaseModel

from kg_construction.data_generation.data_manager.data_storage_models.meta_data import (
    MetaData,
)
from .async_llm_utils import Async_Openai_Compatible
from .async_llm_utils import get_completion_list
from .sync_llm_utils import LLM


def single_generation_with_llm_csv(
    prompt: Dict[str, str],
    prompt_config: Dict[str, Any],
    schema_file: str,
    model: str = "llama3.1-70b",
    seed: int = None,
    temperature: float = 0.05,
    format_model: Type[BaseModel] = None,
    choices=None,
) -> (str, MetaData):
    meta_data_dict = {
        "dateTime": datetime.now(),
        "prompt": prompt,
        "prompt_config": prompt_config,
        "temperature": temperature,
        "model": model,
        "schema": schema_file,
    }
    meta_data = MetaData(**meta_data_dict)
    prompt = meta_data.assembled_prompt
    llm = LLM()
    if format_model:
        format_schema = format_model.model_json_schema()
        response, response_time = llm.call_with_formater(
            format_schema,
            user_prompt=prompt["user_message"],
            system_prompt=prompt["system_message"],
            model_name=model,
            temperature=temperature,
            seed=seed,
        )
    else:
        if choices is not None:
            response, response_time = llm.call_with_choice(
                user_prompt=prompt["user_message"],
                system_prompt=prompt["system_message"],
                model_name=model,
                temperature=temperature,
                choices=choices,
            )
        else:
            response, response_time = llm.call(
                user_prompt=prompt["user_message"],
                system_prompt=prompt["system_message"],
                model_name=model,
                temperature=temperature,
            )
    meta_data.response_time = response_time
    return response, meta_data


async def async_generation_with_csv(
    prompt_list: List[Dict[str, str]],
    prompt_config_list: List[Dict[str, Any]],
    schema_file: str,
    model: str = "llama3.1-70b",
    temperature: float = 0.05,
    max_parallel_calls=10,
    format_model: Type[BaseModel] = None,
    choices=None,
) -> List:
    meta_data_list = []
    content_list = []
    for prompt, prompt_config in zip(prompt_list, prompt_config_list):
        meta_data_dict = {
            "dateTime": datetime.now(),
            "prompt": prompt,
            "prompt_config": prompt_config,
            "temperature": temperature,
            "model": model,
            "schema": schema_file,
        }
        meta_data = MetaData(**meta_data_dict)
        meta_data_list.append(meta_data)
        complete_prompt = meta_data.assembled_prompt
        content_list.append(complete_prompt)
    llm = Async_Openai_Compatible(model=model, temperature=temperature)
    if format_model:
        format_model = format_model.model_json_schema()
    answers = await get_completion_list(
        content_list,
        llm,
        max_parallel_calls,
        choice_enforcing=choices,
        format_enforcing=format_model,
    )
    return list(zip(answers, meta_data_list))
