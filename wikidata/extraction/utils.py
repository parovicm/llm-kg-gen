import os
import re
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

sys.path.append("/ccs_data/marinela/projects/llm-kg-gen")

from simple_wikidata_db.utils import jsonl_generator, get_batch_files
from simple_wikidata_db.preprocess_utils.writer_process import Table
from sparql_api_utils import get_all_subclasses_for_item


def get_relations(pids, filename, keep_types=[]):
    filtered = []
    for item in jsonl_generator(filename):
        if item["property_id"] in pids and (
            item["qid"] in keep_types or (len(keep_types) == 0)
        ):
            del item["claim_id"]
            filtered.append(item)
    return filtered


def get_all_labels(filename):
    items = {}
    for item in jsonl_generator(filename):
        items[item["qid"]] = item["label"]
    return items


def get_labels(triples, labels, keep_qids=None, has_qid=True):
    if keep_qids:
        triples = [
            triple
            for triple in triples
            if triple["qid"] in keep_qids or triple["value"] in keep_qids
        ]
    triples_qids = set()
    for row in triples:
        triples_qids.add(row["qid"])
        if has_qid:
            triples_qids.add(row["value"])
    labels_filtered = {key: labels[key] for key in triples_qids if key in labels}
    triples_updated = []
    for row in triples:
        row["qid_label"] = labels_filtered.get(row["qid"], "")
        row["value_label"] = (
            labels_filtered.get(row["value"], "") if has_qid else row["value"]
        )
        if row["qid_label"] != "":
            triples_updated.append(row)
    return triples_updated


def get_aliases(filename):
    items = defaultdict(list)
    for item in jsonl_generator(filename):
        items[item["qid"]].append(item["alias"])
    return items


def extract_relation(
    data_dir,
    output_dir,
    pid,
    main_qid=None,
    add_labels=True,
    has_qid=True,
    num_procs=10,
):
    keep_qids = (
        set(pd.read_csv(os.path.join(output_dir, f"{main_qid}.csv"))["qid"])
        if main_qid
        else set()
    )

    pool = Pool(processes=num_procs)
    table_files = get_batch_files(
        os.path.join(data_dir, "entity_rels" if has_qid else "entity_values")
    )
    filtered = []
    for output in tqdm(
        pool.imap_unordered(partial(get_relations, [pid]), table_files, chunksize=1),
        total=len(table_files),
    ):
        filtered.extend(output)

    if add_labels:
        labels_files = get_batch_files(os.path.join(data_dir, "labels"))
        labels = {}
        for output in tqdm(
            pool.imap_unordered(get_all_labels, labels_files, chunksize=1),
            total=len(labels_files),
        ):
            labels.update(output)
        filtered = get_labels(filtered, labels, keep_qids=keep_qids, has_qid=has_qid)
    else:
        if keep_qids:
            filtered = [
                triple
                for triple in filtered
                if triple["qid"] in keep_qids or triple["value"] in keep_qids
            ]
    print(f"Extracted {len(filtered)} rows.")

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    table = Table(Path(output_dir), 50000, f"{pid}")
    table.write(filtered)


def filter_by_classes(filename, classes):
    df = pd.read_json(filename, lines=True)
    df = df[["qid", "value"]]
    df = df[df["value"].isin(classes)]
    return df


def load_subclasses_instances(
    qid,
    subclasses_dir="/ccs_data/marinela/subclasses",
    data_dir="/ccs_data/marinela/test_wiki",
    wikidata_dir="/ccs_data/marinela/wikidata_dump/processed_50000_en",
    output_dir="/ccs_data/marinela/test_wiki",
    num_procs=10,
):
    subclasses_list = pd.read_csv(os.path.join(subclasses_dir, f"{qid}.csv"))[
        "item"
    ].tolist()
    pattern = r"Q\d+"
    subclasses_list = [re.search(pattern, value).group() for value in subclasses_list]
    filtered_instance_of = filter_by_classes(
        os.path.join(data_dir, "P31/0.jsonl"), subclasses_list
    )

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    filtered_instance_of.to_csv(os.path.join(output_dir, f"{qid}.csv"))

    labels_files = get_batch_files(os.path.join(wikidata_dir, "labels"))
    pool = Pool(processes=num_procs)
    labels = {}
    for output in tqdm(
        pool.imap_unordered(get_all_labels, labels_files, chunksize=1),
        total=len(labels_files),
    ):
        labels.update(output)

    filtered_with_labels = get_labels(
        filtered_instance_of.to_dict(orient="records"), labels
    )
    table = Table(Path(output_dir), 50000, f"{qid}_labels")
    table.write(filtered_with_labels)
    return filtered_with_labels


def extract_aliases(
    data_dir, output_dir, main_qid=None, rel=None, entity_name=None, num_procs=10
):
    if rel:
        keep_qids = set(
            [
                val["value"]
                for val in jsonl_generator(os.path.join(output_dir, f"{rel}/0.jsonl"))
            ]
        )
    else:
        keep_qids = set(
            pd.read_csv(os.path.join(output_dir, f"{main_qid}.csv"))["qid"].tolist()
        )

    alias_files = get_batch_files(os.path.join(data_dir, "aliases"))
    pool = Pool(processes=num_procs)
    labels = {}
    for output in tqdm(
        pool.imap_unordered(get_aliases, alias_files, chunksize=1),
        total=len(alias_files),
    ):
        labels.update(output)
    labels_filtered = [{key: labels[key]} for key in labels if key in keep_qids]
    table = Table(
        Path(os.path.join(output_dir, "alias")), 50000, rel if rel else entity_name
    )
    table.write(labels_filtered)


# Load jsonl file with lines like:
# {"qid":"Q170583","value":"Q7725634","qid_label":"Pride and Prejudice","value_label":"literary work"}
# and convert it into a csv file with name <qid>.csv and columns item, itemLabel
# /ccs_data/marinela/written_work/Q47461344_labels/0.jsonl
def jsonl_to_csv_item_label(data_path, qid):
    result = []
    for record in jsonl_generator(os.path.join(data_path, f"{qid}_labels/0.jsonl")):
        result.append({"item": record["qid"], "itemLabel": record["qid_label"]})
    pd.DataFrame(result).drop_duplicates().to_csv(
        os.path.join(data_path, "tmp", f"{qid}.csv"), index=False
    )


# alias (/ccs_data/marinela/written_work/alias/<book or relation PID>/0.jsonl):
# {"Q108282612":["1047 Games","1047 Games, Inc."]}
# jsonl for book (/ccs_data/marinela/written_work/Q47461344_labels/0.jsonl):
# {"qid":"Q43236132","value":"Q871232","qid_label":"Pride and prejudice", "value_label": "editorial"}
# jsonl for other relations contains lines like these (/ccs_data/marinela/written_work/<PID>/0.jsonl):
# {"qid":"Q28002560","property_id":"P123","value":"Q11281443","qid_label":"Pride and Prejudice","value_label":"Penguin Classics"}
def jsonl_to_csv_label_alias(data_path, pids, entity):
    qid_label = {}
    result = []
    for pid in pids:
        for record in jsonl_generator(os.path.join(data_path, pid, "0.jsonl")):
            qid_label[record["value"]] = record["value_label"]
    aliases = {}
    for pid in pids:
        for record in jsonl_generator(os.path.join(data_path, "alias", pid, "0.jsonl")):
            for key, value in record.items():
                aliases[key] = value
        for qid, label in qid_label.items():
            if qid not in aliases:
                result.append({"itemLabel": label, "alias": ""})
            else:
                for alias in aliases[qid]:
                    result.append({"itemLabel": label, "alias": alias})
    os.makedirs(os.path.join(data_path, "tmp"), exist_ok=True)
    pd.DataFrame(result).to_csv(
        os.path.join(data_path, "tmp", f"{entity}_label_alias.csv"), index=False
    )


def jsonl_to_csv_label_alias_main(data_path, qid, entity):
    qid_label = {}
    result = []
    for record in jsonl_generator(os.path.join(data_path, f"{qid}_labels/0.jsonl")):
        qid_label[record["qid"]] = record["qid_label"]
    aliases = {}
    for record in jsonl_generator(os.path.join(data_path, f"alias/{entity}/0.jsonl")):
        for key, value in record.items():
            aliases[key] = value
    for qid, label in qid_label.items():
        if qid not in aliases:
            result.append({"itemLabel": label, "alias": ""})
        else:
            for alias in aliases[qid]:
                result.append({"itemLabel": label, "alias": alias})
    os.makedirs(os.path.join(data_path, "tmp"), exist_ok=True)
    pd.DataFrame(result).to_csv(
        os.path.join(data_path, f"tmp/{entity}_label_alias.csv"), index=False
    )


def jsonl_to_csv_relation(data_path, pids, entity1, entity2):
    result = []
    for pid in pids:
        for record in jsonl_generator(os.path.join(data_path, pid, "0.jsonl")):
            result.append(
                {entity1: record["qid_label"], entity2: record["value_label"]}
            )
    os.makedirs(os.path.join(data_path, "tmp"), exist_ok=True)
    pd.DataFrame(result).drop_duplicates().to_csv(
        os.path.join(data_path, "tmp", f"{entity1}_{entity2}.csv"), index=False
    )


def extract_wikidata(
    main_qid: str,
    main_entity: str,
    base_output_dir: str,
    subclasses_dir: str,
    data_dir: str,
    pids: List,
    pids_no_qid: List,
    entity_mapping: Dict[str, List],
    relation_mapping: List,
):
    get_all_subclasses_for_item(item_id=main_qid, output_dir=subclasses_dir)
    extract_relation(
        data_dir=data_dir, output_dir=base_output_dir, pid="P31", add_labels=False
    )
    load_subclasses_instances(
        qid=main_qid,
        subclasses_dir=subclasses_dir,
        data_dir=base_output_dir,
        wikidata_dir=data_dir,
        output_dir=base_output_dir,
    )

    for pid in pids:
        print(f"Extracting relation {pid}...")
        extract_relation(
            data_dir=data_dir, output_dir=base_output_dir, pid=pid, main_qid=main_qid
        )

    for pid in pids_no_qid:
        print(f"Extracting relation {pid}...")
        extract_relation(
            data_dir=data_dir,
            output_dir=base_output_dir,
            pid=pid,
            main_qid=main_qid,
            has_qid=False,
        )

    extract_aliases(
        data_dir=data_dir,
        output_dir=base_output_dir,
        main_qid=main_qid,
        entity_name=main_entity,
    )
    for pid in pids:
        print(f"Extracting aliases {pid}...")
        extract_aliases(data_dir=data_dir, output_dir=base_output_dir, rel=pid)
    for pid in pids_no_qid:
        print(f"Making empty alias file for {pid}...")
        os.makedirs(os.path.join(base_output_dir, f"alias/{pid}"), exist_ok=True)
        with open(os.path.join(base_output_dir, f"alias/{pid}/0.jsonl"), "w") as f:
            pass

    for en, pids in entity_mapping.items():
        print(en)
        jsonl_to_csv_label_alias(data_path=base_output_dir, pids=pids, entity=en)

    jsonl_to_csv_label_alias_main(base_output_dir, main_qid, main_entity)

    for rels, en1, en2 in relation_mapping:
        print(en2)
        jsonl_to_csv_relation(
            data_path=base_output_dir, pids=rels, entity1=en1, entity2=en2
        )
