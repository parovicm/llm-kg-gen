import json
from datetime import datetime
from typing import Dict, Any, Optional

from pydantic import BaseModel, model_validator, PlainSerializer
from typing_extensions import Annotated

CustomTime = Annotated[
    datetime,
    PlainSerializer(lambda time: time.strftime("%Y-%m-%d %H:%M:%S"), return_type=str),
]


class MetaData(BaseModel):
    dateTime: CustomTime
    prompt: Dict[str, Any]
    prompt_config: Dict[str, Any]
    temperature: float
    model: str
    schema: str
    assembled_prompt: Dict[str, Any] = {}
    response_time: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def check_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(values, dict):
            config = values.get("prompt_config", {})
            if not isinstance(config, dict):
                raise ValueError("prompt_config must be a dictionary.")
            prompt_data = values.get("prompt")
            try:
                if isinstance(prompt_data, dict):
                    prompt = prompt_data
                else:
                    raise ValueError(
                        "Prompt should be either a string or a dictionary."
                    )
                formatted_prompt = {
                    k: v.format(**config)
                    for k, v in prompt.items()
                    if isinstance(v, str)
                }
                values["assembled_prompt"] = formatted_prompt
            except json.JSONDecodeError:
                raise ValueError("Prompt is not a valid json.")

            except KeyError as e:
                raise ValueError(
                    f"Missing key in config used in prompt formatting: {str(e)}"
                )
            return values


if __name__ == "__main__":
    input_data = {
        "dateTime": datetime.now(),
        "prompt": {"message": "Hello, {name}!"},
        "prompt_config": {"name": "Alice"},
        "temperature": 0.2,
        "model": "llama3.1-70b",
        "schema": "asdf",
    }

    meta_data = MetaData(**input_data)
    print(meta_data.model_dump_json())
