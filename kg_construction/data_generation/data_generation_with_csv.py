import copy
import json
from typing import List

from tqdm import tqdm

from kg_construction.cove import EntityListCOVEChain, CustomOpenAI
from kg_construction.utils.data_generation_utils import (
    single_generation_with_llm_csv,
    async_generation_with_csv,
)
from kg_construction.utils.format_enforcing_utils import (
    create_list_of_object,
    RelationList,
)
from .data_manager.csv_manager import CsvManager


def generate_initial_relations(
    prompt_file: str,
    n: int,
    manager: CsvManager,
    model: str = "llama3.1-70b",
    seed: int = None,
    temperature: float = 0.2,
):
    """
    Start generation by generating n instances for all relations.
    Args:
        schema_file: the path to the file that stores the schema
        prompt_file: the paths to the file that stores the prompt for this call
        n: the number of relation instances that will be generated
        model: the model that is used to generate the instances
        seed: the seed of the call
        temperature: the temperature of the call
        output: the dictionary that is used to store the output

    Returns: ()

    """
    schema_obj = manager.schema
    relations_list = schema_obj.relation_triples
    schema_file = schema_obj.file_name
    config = {"N": n}
    relation_object_list_model = create_list_of_object(
        object_type=RelationList, class_name="relation", n=n, item_name="relation"
    )

    # load the propt json to avoid repetitive loading of a file
    with open(prompt_file, "r", encoding="utf-8") as prompt_data:
        prompt = json.load(prompt_data)

    # traverse through all relations in the schema
    for entity1, entity2, relation in relations_list:
        gen_args = {"entity1": entity1, "entity2": entity2, "relation": relation}
        config |= gen_args
        (response, meta_data) = single_generation_with_llm_csv(
            prompt=prompt,
            prompt_config=config,
            schema_file=schema_file,
            model=model,
            seed=seed,
            temperature=temperature,
            format_model=relation_object_list_model,
            choices=None,
        )
        print(response)
        manager.turn_relation_into_entity(
            response, relation, item_name="entity", metadata=meta_data
        )
        manager.write_data(relation)


async def async_generate_initial_relations(
    prompt_file: str,
    n: int,
    manager: CsvManager,
    model: str = "llama3.1-70b",
    seed: int = None,
    temperature: float = 0.2,
):
    """
    Start generation by generating n instances for all relations.
    Args:
        schema_file: the path to the file that stores the schema
        prompt_file: the paths to the file that stores the prompt for this call
        n: the number of relation instances that will be generated
        model: the model that is used to generate the instances
        seed: the seed of the call
        temperature: the temperature of the call
        output: the dictionary that is used to store the output

    Returns: ()

    """
    schema_obj = manager.schema
    relations_list = schema_obj.relation_triples
    schema_file = schema_obj.file_name
    relation_object_list_model = create_list_of_object(
        object_type=RelationList, class_name="relation", n=n, item_name="relation"
    )

    # load the prompt json to avoid repetitive loading of a file
    with open(prompt_file, "r", encoding="utf-8") as prompt_data:
        prompt = json.load(prompt_data)

    prompt_list = []
    prompt_config_list = []
    # traverse through all relations in the schema
    for entity1, entity2, relation in relations_list:
        gen_args = {"entity1": entity1, "entity2": entity2, "relation": relation}
        config = {"N": n}
        config |= gen_args
        config_copy = copy.deepcopy(config)
        prompt_copy = copy.deepcopy(prompt)
        prompt_list.append(prompt_copy)
        prompt_config_list.append(config_copy)

    zipped_list = await async_generation_with_csv(
        prompt_list=prompt_list,
        prompt_config_list=prompt_config_list,
        schema_file=schema_file,
        model=model,
        temperature=temperature,
        max_parallel_calls=10,
        format_model=relation_object_list_model,
    )
    for (response, meta_data), relation in zip(zipped_list, relations_list):
        manager.turn_relation_into_entity(
            response, relation[2], item_name="entity", metadata=meta_data
        )
    manager.write_all_data()


def complete_entities(
    prompt_file: str,
    manager: CsvManager,
    model: str = "llama3.1-70b",
    seed: int = None,
    temperature: float = 0.2,
):
    schema_object = manager.schema
    main_entity = schema_object.main_entity
    config = {"schema": str(schema_object.relationless_schema)}
    schema_class = schema_object.pydantic_class
    data = manager.incomplete_relations
    visited = set()

    with open(prompt_file, "r", encoding="utf-8") as prompt_data:
        prompt = json.load(prompt_data)

    for source_en, target_en, rel in schema_object.relation_triples:
        relation_objects: List[List] = data[rel]
        # get the values of the other entity in a relation
        for relation in relation_objects:
            source_val = relation[1]
            target_val = relation[2]
            if source_en == main_entity:
                if source_val in visited:
                    continue
                visited.add(source_val)
            elif target_en == main_entity:
                if target_val in visited:
                    continue
                visited.add(target_val)
            gen_args = {
                "entity1": source_en,
                "entity2": target_en,
                "value1": source_val,
                "value2": target_val,
            }
            config |= gen_args
            response, metaData = single_generation_with_llm_csv(
                prompt=prompt,
                prompt_config=config,
                schema_file=schema_object.file_name,
                model=model,
                seed=seed,
                temperature=temperature,
                format_model=schema_class,
            )
            print(response)
            manager.create_entity_and_relation_from_schema(response, metaData)
            manager.write_all_data()


async def async_complete_entities(
    prompt_file: str,
    manager: CsvManager,
    model: str = "llama3.1-70b",
    seed: int = None,
    temperature: float = 0.2,
):
    schema_object = manager.schema
    main_entity = schema_object.main_entity
    config = {"schema": str(schema_object.relationless_schema)}
    data = manager.incomplete_relations
    visited = set()

    with open(prompt_file, "r", encoding="utf-8") as prompt_data:
        prompt = json.load(prompt_data)

    for source_en, target_en, rel in schema_object.relation_triples:
        print(f"Relation: {rel}")
        relation_objects: List[List] = data[rel]
        # get the values of the other entity in a relation
        prompt_list = []
        prompt_config_list = []
        for relation in relation_objects:
            source_val = relation[1]
            target_val = relation[2]
            if source_en == main_entity:
                if source_val in visited:
                    continue
                visited.add(source_val)
            elif target_en == main_entity:
                if target_val in visited:
                    continue
                visited.add(target_val)
            gen_args = {
                "entity1": source_en,
                "entity2": target_en,
                "value1": source_val,
                "value2": target_val,
            }
            config |= gen_args
            config_copy = copy.deepcopy(config)
            prompt_copy = copy.deepcopy(prompt)
            prompt_list.append(prompt_copy)
            prompt_config_list.append(config_copy)

        zipped_list = await async_generation_with_csv(
            prompt_list=prompt_list,
            prompt_config_list=prompt_config_list,
            schema_file=schema_object.file_name,
            model=model,
            temperature=temperature,
            max_parallel_calls=5,
            format_model=schema_object.pydantic_class,
        )
        for response, meta_data in zipped_list:
            manager.create_entity_and_relation_from_schema(response, meta_data)
        manager.write_all_data()


def generate_entities(
    prompt_file: str,
    manager: CsvManager,
    n: int,
    model: str = "llama3.1-70b",
    seed: int = None,
    temperature: float = 0.2,
    output=None,
):
    """
    Generate more main entities based on already generated entities.
    Args:
        prompt_file:
        manager:
        n:
        model:
        seed:
        temperature:
        output:

    Returns:

    """
    hallucination = ["Error", "Mistake", "Fictional", "Hypothetical", "Uncertain"]
    gen_entity = manager.schema.main_entity
    config = {"N": n, "entity": gen_entity}
    with open(prompt_file, "r", encoding="utf-8") as prompt_data:
        prompt = json.load(prompt_data)
    data = manager.incomplete_relations
    cove_llm = CustomOpenAI(
        model_name=model,
        **{"temperature": temperature, "seed": seed, "max_retries": 20},
    )
    for en1, en2, rel in manager.schema.relation_triples:
        manager.set_relation_complete(rel)
        relation_objects: List[List] = data[rel]
        for r in relation_objects:
            if gen_entity == en1:
                target_entity = en2
                target_entity_val = r[1]
                input_entity_val = r[2]
            else:
                target_entity = en1
                target_entity_val = r[2]
                input_entity_val = r[1]
            if (not any(sub in input_entity_val for sub in hallucination)) and (
                not any(sub in target_entity_val for sub in hallucination)
            ):
                gen_args = {
                    "input_entity": target_entity,
                    "input_entity_value": input_entity_val,
                    "example": target_entity_val,
                }
                config |= gen_args
                response, metadata = single_generation_with_llm_csv(
                    prompt=prompt,
                    prompt_config=config,
                    schema_file=manager.schema.file_name,
                    model=model,
                    seed=seed,
                    temperature=temperature,
                )
                cove_chain_instance = EntityListCOVEChain(
                    cove_llm, baseline_response_exists=True
                )
                cove_chain = cove_chain_instance()
                cove_chain_result = cove_chain.invoke(
                    {
                        "original_question": prompt["user_message"],
                        "baseline_response": response,
                    }
                )
                response = cove_chain_result["final_answer"]
                print(response)
                manager.get_generate_more_entities(
                    response, input_entity_val, (en1, en2, rel), metadata
                )
        manager.write_data(rel)


async def async_generate_entities(
    prompt_file: str,
    manager: CsvManager,
    n: int,
    use_cove: bool = True,
    model: str = "llama3.1-70b",
    seed: int = None,
    temperature: float = 0.2,
    output=None,
):
    """
    Generate more main entities based on already generated entities.
    Args:
        prompt_file:
        manager:
        n:
        use_cove:
        model:
        seed:
        temperature:
        output:

    Returns:

    """
    hallucination = ["Error", "Mistake", "Fictional", "Hypothetical", "Uncertain"]
    gen_entity = manager.schema.main_entity
    config = {"N": n, "entity": gen_entity}
    with open(prompt_file, "r", encoding="utf-8") as prompt_data:
        prompt = json.load(prompt_data)
    data = manager.incomplete_relations
    cove_llm = CustomOpenAI(
        openai_api_key="sk-1234",
        openai_api_base="http://10.227.91.60:4000/v1",
        model_name=model,
        **{
            "max_tokens": 2048,
            "temperature": temperature,
            "seed": seed,
            "max_retries": 20,
        },
    )
    for en1, en2, rel in manager.schema.relation_triples:
        print(f"Relation: {rel}")
        relation_objects: List[List] = data[rel]
        prompt_list = []
        prompt_config_list = []
        input_entity_val_list = []
        for r in relation_objects:
            if gen_entity == en1:
                target_entity = en2
                target_entity_val = r[1]
                input_entity_val = r[2]
            else:
                target_entity = en1
                target_entity_val = r[2]
                input_entity_val = r[1]
            if (not any(sub in input_entity_val for sub in hallucination)) and (
                not any(sub in target_entity_val for sub in hallucination)
            ):
                gen_args = {
                    "input_entity": target_entity,
                    "input_entity_value": input_entity_val,
                    "example": target_entity_val,
                }
                config |= gen_args
                config_copy = copy.deepcopy(config)
                prompt_copy = copy.deepcopy(prompt)
                prompt_list.append(prompt_copy)
                prompt_config_list.append(config_copy)
                input_entity_val_list.append(input_entity_val)
        zipped_list = await async_generation_with_csv(
            prompt_list=prompt_list,
            prompt_config_list=prompt_config_list,
            schema_file=manager.schema.file_name,
            model=model,
            temperature=temperature,
            max_parallel_calls=5,
        )
        if use_cove:
            COVE_reponse_list = []
            for (response, meta_data), prompt in tqdm(zip(zipped_list, prompt_list)):
                assembled_prompt = meta_data.assembled_prompt["user_message"]
                cove_chain_instance = EntityListCOVEChain(
                    cove_llm, baseline_response_exists=True, max_parallel_calls=10
                )
                cove_chain = cove_chain_instance()
                cove_chain_result = await cove_chain.ainvoke(
                    {
                        "original_question": assembled_prompt,
                        "baseline_response": response,
                    }
                )
                final_response = cove_chain_result["final_answer"]
                COVE_reponse_list.append((final_response, meta_data))
            zipped_list = COVE_reponse_list
        for (response, meta_data), input_entity_val in zip(
            zipped_list, input_entity_val_list
        ):
            manager.get_generate_more_entities(
                response, input_entity_val, (en1, en2, rel), meta_data
            )
        manager.set_relation_complete(rel)
        manager.write_data(rel)
