from typing import Dict, Any, List, Type

from pydantic import BaseModel, create_model


def create_pydantic_class_from_dict(
    name: str, fields: Dict[str, Any]
) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic class from a given set of fields.

    Args:
    - name (str): Name of the class to create.
    - fields (dict): Dictionary of fields where keys are attribute names and values are their types.

    Returns:
    - A new Pydantic model class.
    """
    schema_dict = {}
    type_mapping = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "List[str]": List[str],
        "string": str,
        "integer": int,
    }

    for field_name, field_type in fields.items():
        if isinstance(field_type, dict):
            # Handle nested structures or complex types
            nested_class_name = f"{name}_{field_name}"
            nested_class = create_pydantic_class_from_dict(
                nested_class_name, field_type
            )
            schema_dict[field_name] = nested_class
        elif isinstance(field_type, list):
            # Assuming list types are properly formatted like ["str"]
            list_type = field_type[0]
            python_list_type = type_mapping.get(list_type, str)
            schema_dict[field_name] = (List[python_list_type], ...)
        else:
            # Simple types directly from the type_mapping
            schema_dict[field_name] = (type_mapping.get(field_type, str), ...)

    # Create the Pydantic model class with the annotations
    return create_model(__model_name=name, **schema_dict, __base__=(BaseModel,))


def create_container_class_for_multi_classes(
    container_name: str, pydantic_classes: Dict[str, Any]
) -> Type[BaseModel]:
    """
    Creates a Pydantic class that contains all the given Pydantic classes.

    Args:
    - class_name (str): The name of the container class to create.
    - pydantic_classes (dict): A dictionary of class names to Pydantic classes.

    Returns:
    - A Pydantic model class that contains all the given classes.
    """
    container = {}
    for name, cls in pydantic_classes.items():
        container[name] = (cls, ...)
    return create_model(container_name, **container, __base__=(BaseModel,))


# Returns a class object not a class instance therefore it is a Type[BaseModel]
def create_pydantic_classes_from_schema(
    schema: Dict[str, Any], class_name
) -> Type[BaseModel]:
    """
    Generates Pydantic classes for all entities defined in a schema.

    Args:
    - schema (dict): The schema dictionary containing entity definitions.

    Returns:
    - A dictionary of class names to Pydantic classes.
    """
    classes = {}
    for entity, details in schema.get("concepts", {}).items():
        class_name = entity
        fields = details.get("attributes", {})
        classes[class_name] = create_pydantic_class_from_dict(class_name, fields)
    return create_container_class_for_multi_classes(class_name, classes)


def create_list_of_object(
    object_type: Type, class_name: str, n: int, item_name: str
) -> Type[BaseModel]:
    object_dict = {f"{item_name}_{i}": (object_type, ...) for i in range(1, n + 1)}
    return create_model(__model_name=class_name, **object_dict, __base__=(BaseModel,))


class RelationList(BaseModel):
    entity1: str
    entity2: str


class EntityList(BaseModel):
    name: str
