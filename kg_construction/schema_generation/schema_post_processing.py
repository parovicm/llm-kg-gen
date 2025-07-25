import copy
import json
import os


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
        return schema


def save_json_from_dict(schema, path):
    dir_path, original_filename = os.path.split(path)
    name, ext = os.path.splitext(original_filename)
    new_filename = f"{name}_processed_categories{ext}"
    new_file_path = os.path.join(dir_path, new_filename)
    # print(new_file_path)
    with open(new_file_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=4, ensure_ascii=False)
        print(f"JSON extracted and saved to: {new_file_path}")


def remove_duplication(schema):
    entities = schema.get("concepts", {})
    keys_to_remove = {}

    for key in entities.keys():
        attributes = entities[key].get("attributes", {})
        keys_to_remove[key] = [
            attr_name
            for attr_name in attributes.keys()
            if ("ID" in attr_name or attr_name in entities.keys())
        ]

    print(f"Removing attributes: {keys_to_remove}")

    for entity_key, attr_keys in keys_to_remove.items():
        for attr_key in attr_keys:
            del schema["concepts"][entity_key]["attributes"][attr_key]

    # remove concepts without attributes
    concepts = schema.get("concepts", {})
    updated_concepts = copy.deepcopy(concepts)
    for concept, details in concepts.items():
        if "attributes" not in details or len(details["attributes"]) == 0:
            del updated_concepts[concept]
        else:
            updated_concepts[concept]["attributes"][
                "categories"
            ] = '["string"]'  # "string"
    schema["concepts"] = updated_concepts

    # remove relations without source, target and name
    relations = schema.get("relations", {})
    updated_relations = copy.deepcopy(relations)
    for relation, relation_data in relations.items():
        if not all(key in relation_data for key in ["name", "source", "target"]):
            del updated_relations[relation]
    schema["relations"] = updated_relations

    # remove relations from x to x and relations with non-existing entities
    relations = copy.deepcopy(schema.get("relations", {}))
    entities = schema.get("concepts", {})
    for relation in relations:
        if relations[relation]["source"] == relations[relation]["target"]:
            del schema["relations"][relation]
        elif (
            relations[relation]["source"] not in entities
            or relations[relation]["target"] not in entities
        ):
            del schema["relations"][relation]

    return schema


def check_for_entity_in_relations(schema):
    entities = schema.get("concepts", {})
    relations = schema.get("relations", {})

    # Check if there are multiple entities
    if len(entities) < 2:
        raise Exception("The schema must contain at least two entities.")

    # Check if there is at least one relation
    if len(relations) < 1:
        raise Exception("The schema must contain at least one relation.")

    entity_relation_counts = {entity: 0 for entity in entities}

    for relation, relation_data in relations.items():
        sources = relation_data.get("source", [])
        if isinstance(sources, str) or not isinstance(sources, list):
            sources = [sources]

        targets = relation_data.get("target", [])
        if isinstance(targets, str) or not isinstance(targets, list):
            targets = [targets]

        # Ensure there is only one source and one target
        if len(sources) != 1 or len(targets) != 1:
            raise Exception(f"Relation '{relation}' is not one-to-one.")

        source = sources[0]
        target = targets[0]

        # Count occurrences for each source and target
        if source in entity_relation_counts:
            entity_relation_counts[source] += 1
        else:
            raise Exception(
                "This source " + source + " for relation " + relation + " is not found"
            )

        if target in entity_relation_counts:
            entity_relation_counts[target] += 1
        else:
            raise Exception(
                "This target " + target + " for relation " + relation + " is not found"
            )

    # Print out how many relations each entity appeared in
    for entity, count in entity_relation_counts.items():
        print(f"Entity '{entity}' appeared in {count} relations.")

    # Ensure main entity appears in at least one relation
    main_entity = next(iter(entities))  # Assuming the first entity is the main one
    if entity_relation_counts[main_entity] >= 1:
        print(f"Main entity '{main_entity}' appeared in at least one relation.")
        return schema
    else:
        raise Exception(f"Main entity '{main_entity}' did not appear in any relations.")


def check_schema_structure(schema):
    # Check top-level keys
    top_keys = set(schema.keys())
    expected_top_keys = {"concepts", "relations"}

    if top_keys != expected_top_keys:
        raise Exception(
            "Error: Schema must have 'concepts' and 'relations' as top-level keys."
        )

    # Check internal structure of "concepts"
    concepts = schema.get("concepts", {})
    for concept, details in concepts.items():
        if "attributes" not in details:
            raise Exception(f"Error: Concept '{concept}' lacks 'attributes'.")

    # Check internal structure of "relations"
    relations = schema.get("relations", {})
    for relation, relation_data in relations.items():
        if not all(key in relation_data for key in ["name", "source", "target"]):
            raise Exception(
                f"Error: Relation '{relation}' lacks one of 'name', 'source', or 'target'."
            )
    print("Schema structure is correct.")
    return schema


def post_processing(json_paths):
    try:
        for json_path in json_paths:
            schema_dic = load_json(json_path)
            schema1 = remove_duplication(schema_dic)
            schema2 = check_for_entity_in_relations(schema1)
            schema3 = check_schema_structure(schema2)
            save_json_from_dict(schema3, json_path)
            if len(json_paths) == 1:
                return schema3
    except:
        print("Invalid")


if __name__ == "__main__":
    post_processing(
        ["/ccs_data/marinela/projects/llm-kg-gen/schema/books/book_schema.json"]
    )
