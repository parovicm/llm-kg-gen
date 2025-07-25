import argparse
import os
import sys
from collections import defaultdict

import pandas as pd
import yaml

sys.path.append("/ccs_data/marinela/projects/llm-kg-gen")

from kg_construction.data_generation.data_manager.csv_manager import CsvManager
from kg_construction.data_generation.data_manager.schema_interface import SchemaEnglish
from utils import evaluate_entities, evaluate_relations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    schema_obj = SchemaEnglish(config["schema_file"])
    manager = CsvManager(schema_obj, config["output"], config["domain"])
    data_dir = config["data_dir"]
    mapping = config["mapping"]
    skip_entities = config.get("skip_entities", [])
    main_entity = schema_obj.main_entity

    # ===================== Entity Evaluation ======================
    entities = defaultdict(set)
    for entity in schema_obj.entities:
        if entity in skip_entities:
            continue
        print(entity)
        if entity == main_entity:
            entities_no_main = [
                ent
                for ent in schema_obj.entities
                if ent != main_entity and ent not in skip_entities
            ]
            for other_entity in entities_no_main:
                entities[entity] |= set(
                    pd.read_csv(f"{data_dir}/{main_entity}_{other_entity}.csv")[
                        entity
                    ].tolist()
                )
        else:
            entities[entity] |= set(
                pd.read_csv(f"{data_dir}/{main_entity}_{entity}.csv")[entity].tolist()
            )
        wikidata_labels_old = pd.read_csv(
            os.path.join(f"{data_dir}/{entity}_label_alias.csv")
        )
        wikidata_labels = (
            pd.concat([wikidata_labels_old["itemLabel"], wikidata_labels_old["alias"]])
            .reset_index(drop=True)
            .squeeze()
            .to_list()
        )
        entities[entity] |= set(wikidata_labels)

    evaluate_entities(
        schema_obj=schema_obj,
        manager=manager,
        entities=entities,
        mapping=mapping,
        skip_entities=skip_entities,
    )

    # ===================== Alias Construction ======================
    wikidata_aliases = {}
    for entity in schema_obj.entities:
        if entity in skip_entities:
            continue
        wikidata_aliases[entity] = (
            pd.read_csv(os.path.join(f"{data_dir}/{entity}_label_alias.csv"))
            .groupby("itemLabel")["alias"]
            .apply(set)
            .reset_index()
        )
        wikidata_aliases[entity] = wikidata_aliases[entity].rename(
            columns={"itemLabel": entity}
        )

    # ===================== Relation Evaluation ======================
    relations = {}
    for rel in schema_obj.relations:
        print(rel)
        en1, en2 = schema_obj.find_relation(rel)
        second_entity = en1 if en2 == schema_obj.main_entity else en2
        if second_entity in skip_entities:
            continue
        relations[rel] = pd.read_csv(
            f"{data_dir}/{schema_obj.main_entity}_{second_entity}.csv"
        ).drop_duplicates()

    evaluate_relations(
        schema_obj=schema_obj,
        manager=manager,
        relations=relations,
        wikidata_aliases=wikidata_aliases,
        skip_entities=skip_entities,
    )


if __name__ == "__main__":
    main()
