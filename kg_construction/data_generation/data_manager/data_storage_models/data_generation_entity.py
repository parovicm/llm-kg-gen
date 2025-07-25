import hashlib
from datetime import datetime
from typing import Dict, Any, Optional

from pydantic import BaseModel, model_serializer, model_validator

from .meta_data import MetaData


class Entity(BaseModel):
    name: str
    attributes: Optional[Dict[str, Any]]
    metaData: MetaData
    id: Optional[str]

    @model_serializer()
    def serialize_model(self):
        res = {"id": self.id, "name": self.name, "metaData": self.metaData}
        if self.attributes is not None:
            for k, v in self.attributes.items():
                res[k] = v
            return res
        else:
            return res

    @model_validator(mode="before")
    @classmethod
    def generate_id(cls, values):
        name = values.get("name", "")
        attributes = values.get("attributes", {})
        attributes_str = "".join(
            f"{k}{v}" for k, v in attributes.items() if k not in ["categories", "tags"]
        )
        concat_str = (name + attributes_str)[:50]
        # Generate a UUID based on the first 50 characters of the concatenated string
        values["id"] = str(hashlib.sha256(concat_str.encode("utf-8")).hexdigest())
        return values


class Relation(BaseModel):
    """when using relation, the first entity of the entity is seen as the primary key and used as source and target."""

    name: str
    source: str
    target: str


class RelationStore(BaseModel):
    """when using relation, the first entity of the entity is seen as the primary key and used as source and target."""

    name: str
    source: str
    target: str
    metaData: MetaData
    completed: bool
    id: Optional[str]

    @model_validator(mode="before")
    @classmethod
    def generate_id(cls, values):
        name = values.get("name", "")
        source = values.get("source", "")
        target = values.get("target", "")
        concat_str = (name + source + target)[:50]
        # Generate a UUID based on the first 50 characters of the concatenated string
        values["id"] = str(hashlib.sha256(concat_str.encode("utf-8")).hexdigest())
        return values


if __name__ == "__main__":
    metadata_input_data = {
        "dateTime": datetime.now(),
        "prompt": {"message": "Hello, {name}!"},
        "prompt_config": {"name": "Alice"},
        "temperature": 0.2,
        "completed": True,
        "model": "llama3.1-70b",
        "schema": "asdf",
    }

    meta_data = MetaData(**metadata_input_data)

    entity_input_data = {"name": "Person", "attributes": None, "metaData": meta_data}

    entity = Entity(**entity_input_data)
    print(entity.model_dump())
