import re
import sys
from typing import Dict, List

import pandas as pd

sys.path.append("/ccs_data/marinela/projects/llm-kg-gen")

from kg_construction.data_generation.data_manager.csv_manager import CsvManager

from kg_construction.data_generation.data_manager.schema_interface import SchemaEnglish


def evaluate_entities(
    schema_obj: SchemaEnglish,
    manager: CsvManager,
    entities: Dict[str, set],
    mapping: Dict[str, str],
    skip_entities: List[str] = None,
):
    total_wikidata_unique = 0
    total_generated_unique = 0
    total_overlap = 0
    results = []
    if skip_entities is None:
        skip_entities = []

    for entity in schema_obj.entities:
        if entity in skip_entities:
            continue
        wikidata_labels = entities[entity]
        generated_labels = manager.nodes_dict[entity][mapping[entity]].to_list()

        wikidata_labels = set(
            [
                re.sub(r"\s+", " ", label.lower().strip()) if not pd.isna(label) else ""
                for label in wikidata_labels
            ]
        )
        generated_labels = set(
            [
                re.sub(r"\s+", " ", label.lower().strip()) if not pd.isna(label) else ""
                for label in generated_labels
            ]
        )

        wikidata_unique = len(wikidata_labels)
        generated_unique = len(generated_labels)
        overlap = len(wikidata_labels & generated_labels)
        print(
            f"{wikidata_unique}\t{generated_unique}\t{overlap}\t{overlap / generated_unique}"
        )
        results.append(
            {
                "Entity": entity,
                "Wikidata": wikidata_unique,
                "Generated": generated_unique,
                "Overlap": overlap,
                "Percentage": overlap / generated_unique,
            }
        )
        total_wikidata_unique += wikidata_unique
        total_generated_unique += generated_unique
        total_overlap += overlap
    print(
        f"{total_wikidata_unique}\t{total_generated_unique}\t{total_overlap}\t{total_overlap / total_generated_unique}"
    )
    results.append(
        {
            "Entity": "Total",
            "Wikidata": total_wikidata_unique,
            "Generated": total_generated_unique,
            "Overlap": total_overlap,
            "Percentage": total_overlap / total_generated_unique,
        }
    )

    # filename = os.path.join(output_dir, "wikidata_entity_book.xlsx")
    # if not os.path.exists(filename):
    #     pd.DataFrame().to_excel(filename, index=False)
    # df = pd.DataFrame.from_dict(results)
    # with pd.ExcelWriter(filename, mode="a", engine="openpyxl") as writer:
    #     df.to_excel(writer, sheet_name=domain, index=False)

    return


def _aliases_as_rows(df, main_column, alias_column):
    expanded_rows = []
    add_columns = [c for c in df.columns if c not in [main_column, alias_column]]
    for index, row in df.iterrows():
        expanded_rows.append([row[main_column]] + [row[col] for col in add_columns])

        if isinstance(row[alias_column], float):
            continue
        # Add each alias as a new row, keeping the same values for other columns
        for alias in row[alias_column]:
            if pd.isna(alias):
                continue
            expanded_rows.append([alias] + [row[col] for col in add_columns])

    # Create a new DataFrame from the expanded rows
    expanded_df = pd.DataFrame(
        expanded_rows, columns=[main_column] + [col for col in add_columns]
    )
    return expanded_df


def evaluate_relations(
    schema_obj: SchemaEnglish,
    manager: CsvManager,
    relations: Dict[str, pd.DataFrame],
    wikidata_aliases: Dict[str, pd.DataFrame],
    skip_entities: List[str] = None,
):
    results = []
    total_generated_unique = 0
    total_matched = 0
    total_correct = 0
    main_entity = schema_obj.main_entity

    if skip_entities is None:
        skip_entities = []

    for relation in schema_obj.relations:
        en1, en2 = schema_obj.find_relation(relation)
        second_entity = en1 if en2 == main_entity else en2
        if second_entity in skip_entities:
            continue

        generated_relations = manager.relation_dict[relation][
            ["source", "target"]
        ].rename(columns={"source": en1, "target": en2})
        generated_relations = generated_relations.map(
            lambda x: re.sub(r"\s+", " ", x.lower().strip()) if not pd.isna(x) else ""
        ).drop_duplicates()

        wikidata_relations = relations[relation][[main_entity, second_entity]]
        wikidata_relations = wikidata_relations.merge(
            wikidata_aliases[en1], how="left", on=en1
        ).rename(columns={"alias": f"{en1}_alias"})
        wikidata_relations = wikidata_relations.merge(
            wikidata_aliases[en2], how="left", on=en2
        ).rename(columns={"alias": f"{en2}_alias"})
        wikidata_relations = _aliases_as_rows(
            wikidata_relations, main_entity, f"{main_entity}_alias"
        )
        wikidata_relations = _aliases_as_rows(
            wikidata_relations, second_entity, f"{second_entity}_alias"
        )
        wikidata_relations = wikidata_relations.map(
            lambda x: re.sub(r"\s+", " ", x.lower().strip()) if not pd.isna(x) else ""
        )
        wikidata_relations_aggregated = (
            wikidata_relations.groupby(main_entity)[second_entity]
            .apply(set)
            .reset_index()
        )

        merged_df = pd.merge(
            generated_relations,
            wikidata_relations_aggregated,
            on=main_entity,
            how="left",
            suffixes=["_generated", "_wikidata"],
        ).dropna()
        merged_df["is_true"] = merged_df.apply(
            lambda row: row[f"{second_entity}_generated"]
            in row[f"{second_entity}_wikidata"],
            axis=1,
        )
        true_count = merged_df["is_true"].sum()
        print(
            f"{len(generated_relations)}\t{len(merged_df)}\t{true_count}\t{true_count / len(generated_relations)}\t{true_count / len(merged_df)}"
        )

        results.append(
            {
                "Relation": f"{en1}_{en2}",
                "Generated": len(generated_relations),
                "Matched": len(merged_df),
                "Overlap Count": true_count,
                "Overlap": true_count / len(generated_relations),
                "Matched Overlap": true_count / len(merged_df),
            }
        )
        total_generated_unique += len(generated_relations)
        total_matched += len(merged_df)
        total_correct += true_count
    results.append(
        {
            "Relation": f"Total",
            "Generated": total_generated_unique,
            "Matched": total_matched,
            "Overlap Count": total_correct,
            "Overlap": total_correct / total_generated_unique,
            "Matched Overlap": total_correct / total_matched,
        }
    )

    print(
        f"{total_generated_unique}\t{total_matched}\t{total_correct}\t{total_correct / total_generated_unique}\t{total_correct / total_matched}"
    )

    # filename = os.path.join(output_dir, "wikidata_relation_book.xlsx")
    # if not os.path.exists(filename):
    #     pd.DataFrame().to_excel(filename, index=False)
    # df = pd.DataFrame.from_dict(results)
    # with pd.ExcelWriter(filename, mode="a", engine="openpyxl") as writer:
    #     df.to_excel(writer, sheet_name=domain, index=False)

    return
