import datetime
import hashlib
import json
import logging
import os.path
import re
import time
from typing import Dict

import pandas as pd
from pydantic import BaseModel

from .data_storage_models.data_generation_entity import RelationStore, Entity
from .data_storage_models.meta_data import MetaData
from .schema_interface import Schema


def is_string_None(value):
    """
    Check if value is considered None.
    """
    return value in {"None", "null", None, ""}


class CsvManager:
    """
    The CsvManager class acts as an intermediary layer to manage CSV files generated during the data generation process.
    It provides functionalities to manage, store, and retrieve data for various entities and relations defined in a schema.
    """

    def __init__(self, schema: Schema, output_dir: str, domain: str):
        """
        Initialize the CsvManager with the given schema, output directory, and domain.

        Args:
            schema (Schema): The schema that defines the structure of the CSV data.
            output_dir (str): The base directory where CSV files will be stored.
            domain (str): The specific domain of the data being managed.

        Returns:
            None

        When initialising, the Csv manager will have to construct several file paths and essential elements.

        _schema: the schema the Csv data generated will follow
        _domain: the domain of the data generated, this is to configure the path of the files
        _output_domain_dir: the directory where the data is stored
        _csv_paths: the path to the CSV file for each entity/relation type and relation in the system.
        _model_list_dict: the dictionary that stores the intermediate data type for each entity/relation.
        _model_data: the dictionary that is used to track the data generation process and produce insight about the amount of data generated

        """
        self._schema: Schema = schema
        self._domain: str = domain
        self._output_domain_dir = os.path.join(output_dir, domain)
        self._csv_paths = {}
        self._model_list_dict: Dict[str, list[BaseModel]] = {}
        self._model_data = {}
        self.__init_manager_storage__()
        cur_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.logger = logging.getLogger(f"{__name__}.{domain}")
        self.logger.setLevel(logging.INFO)

        log_file = os.path.join(self._output_domain_dir, f"{cur_date_time}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

        self.logger.info("Manager init.")

    @property
    def schema(self) -> Schema:
        """
        Access the schema associated with the CsvManager.

        Args:
            None

        Returns:
            Schema: The schema object.
        """
        return self._schema

    def __init_manager_storage__(self):
        """
        Initialize storage by creating necessary directories and CSV files based on the schema.

        Args:
            None

        Returns:
            None
        """

        os.makedirs(self._output_domain_dir, exist_ok=True)
        for e in self._schema.entities.keys():
            path = os.path.join(self._output_domain_dir, f"{e}.csv")
            self._csv_paths[e] = path
            self._model_list_dict[e] = []
            self._model_data[e] = 0
            if os.path.exists(path):
                continue
            else:
                field = (
                    ["id", "name"]
                    + list(self._schema.get_attribute_for_entity(e).keys())
                    + ["metaData"]
                )
                df = pd.DataFrame(columns=field)
                df.index.name = "index"
                df.to_csv(path)

        for r in self._schema.relations.keys():
            path = os.path.join(self._output_domain_dir, f"{r}.csv")
            self._csv_paths[r] = path
            self._model_list_dict[r] = []
            self._model_data[r] = 0
            if os.path.exists(path):
                continue
            else:
                field = ["id", "name", "source", "target", "metaData", "completed"]
                df = pd.DataFrame(columns=field)
                df.index.name = "index"
                df.to_csv(path)

    def empty_entity_dict(self, entity):
        """
        Reset the list of data for a specific entity type.

        Args:
            entity (str): The name of the entity type to reset.

        Returns:
            None
        """
        if entity not in self._model_list_dict:
            self.logger.error(f"Attempted to reset non-existent entity '{entity}'.")
            raise KeyError(f"Entity '{entity}' does not exist.")
        self._model_list_dict[entity] = []

    def turn_relation_into_entity(
        self, response: str, relation: str, item_name: str, metadata: MetaData
    ):
        """
        Convert generated relations into entities and store them in the CSV files.

        Args:
            response (str): The JSON string response from the model.
            relation (str): The name of the relation being processed.
            item_name (str): The key name used in the response to extract relation items.
            metadata (MetaData): Metadata associated with the data generation process.

        Returns:
            None
        """
        start = time.time()
        count = 0
        data = None
        try:
            data = json.loads(response)
        except json.decoder.JSONDecodeError as e:
            print("Invalid json format")
        if data is None:
            return
        response = data
        meta_data = metadata.model_dump()
        for k, v in response.items():
            source = v[f"{item_name}1"]
            target = v[f"{item_name}2"]
            if (not is_string_None(source)) and (not is_string_None(target)):
                relation_model_dict = {
                    "name": relation,
                    "source": v[f"{item_name}1"],
                    "target": v[f"{item_name}2"],
                    "metaData": meta_data,
                    "completed": False,
                }
                relation_model = RelationStore(**relation_model_dict)
                self._model_list_dict[relation].append(relation_model)
                self._model_data[relation] += 1
                count += 1
        duration = time.time() - start
        logging.info(
            f"Added {count} {relation} s from initial relations, took {duration}"
        )

    def get_generate_more_entities(self, response, secondary_entity, rel, metaData):
        """
        Parse the response from generate_more_entities, filter out hallucinated results, and create relation entities.

        Args:
            response (str): The unstructured response generated by generate_more_entities.
            secondary_entity (str): The name of the secondary entity involved in the relation.
            rel (tuple): A tuple containing relation information (e.g., source entity, target entity, relation name).
            metaData (MetaData): Metadata associated with the data generation process.

        Returns:
            None
        """
        start = time.time()
        count = 0
        main_entity = self.schema.main_entity
        expression = r'\d+\.\s+(?:\*\*|["“”])?([^()\*\n]+)(?:\*\*|["”])?(.*)'
        hallucination = [
            "Error",
            "error",
            "Mistake",
            "mistake",
            "Fictional",
            "fictional",
            "Hypothetical",
            "hypothetical",
            "Uncertain",
            "uncertain",
            "None",
            "none",
        ]
        values = re.findall(expression, response)
        filtered_values = [
            pair[0].strip()
            for pair in values
            if not any(sub in pair[1] for sub in hallucination)
            and pair[0].strip() != "None"
        ]
        for value in filtered_values:
            if rel[0] == main_entity:
                relation_dict = {
                    "name": rel[2],
                    "source": value,
                    "target": secondary_entity,
                    "metaData": metaData.model_dump(),
                    "completed": False,
                }
                entity = RelationStore(**relation_dict)
            else:
                relation_dict = {
                    "name": rel[2],
                    "source": secondary_entity,
                    "target": value,
                    "metaData": metaData.model_dump(),
                    "completed": False,
                }
                entity = RelationStore(**relation_dict)
            self._model_list_dict[rel[2]].append(entity)
            self._model_data[rel[2]] += 1
            count += 1
        duration = time.time() - start
        logging.info(
            f"Added {count} {rel[2]} s in generate more entities, took {duration}"
        )

    def create_entity(self, name: str, metadata: MetaData, attributes=None):
        """
        Create an entity based on the provided name, metadata, and optional attributes.

        Args:
            name (str): The name of the entity to create.
            metadata (MetaData): Metadata associated with the entity.
            attributes (Optional[Dict[str, Optional[str]]]): A dictionary of attribute names and their values.

        Returns:
            Entity: An instance of the Entity model.
        """
        entity_dict = self.schema.entities
        if name not in entity_dict.keys():
            raise ValueError(f"False name for entity: {name}")
        else:
            schema_attribute_names = self._schema.get_attribute_for_entity(name)
            schema_attribute_dict = {k: None for k in schema_attribute_names.keys()}
            if attributes is None:
                entity_dict = {"name": name, "metaData": metadata.model_dump()}
                return Entity(**entity_dict)
            else:
                for k, v in attributes.items():
                    if k in schema_attribute_dict.keys():
                        schema_attribute_dict[k] = v
                entity_dict = {
                    "name": name,
                    "attributes": schema_attribute_dict,
                    "metaData": metadata.model_dump(),
                }
                return Entity(**entity_dict)

    def create_entity_and_relation_from_schema(
        self, schema_json: str, metadata: MetaData
    ):
        """
        Create entities and relations based on the provided schema JSON and metadata.

        Args:
            schema_json (str): A JSON string representing the schema data.
            metadata (MetaData): Metadata associated with the data generation process.

        Returns:
            None
        """
        start = time.time()
        data = None
        try:
            data = json.loads(schema_json)
        except json.decoder.JSONDecodeError as e:
            print("Invalid json format")
        if data is None:
            return
        main = self.schema.main_entity
        for k in data.keys():
            entity_data = data[k]
            primary_key = data[k][list(data[k].keys())[0]]
            if not is_string_None(primary_key):
                entity = self.create_entity(k, metadata, attributes=entity_data)
                self._model_list_dict[k].append(entity)
                self._model_data[k] += 1
                if k != main:
                    value = self.schema.find_relation_between_entities(main, k)
                    if value is None:
                        continue
                    source, target, rel = value
                    relation_object_dict = {
                        "name": rel,
                        "source": data[source][list(data[source].keys())[0]],
                        "target": data[target][list(data[target].keys())[0]],
                        "metaData": metadata.model_dump(),
                        "completed": False,
                    }
                    if (not is_string_None(relation_object_dict["target"])) and (
                        not is_string_None(relation_object_dict["source"])
                    ):
                        self._model_list_dict[rel].append(
                            RelationStore(**relation_object_dict)
                        )
                        self._model_data[rel] += 1
        duration = time.time() - start
        logging.info(
            f"After adding schema, we have {self._model_data}, took {duration}"
        )

    def write_all_data(self):
        """
        Write all cached data for every entity and relation to their respective CSV files.

        Args:
            None

        Returns:
            None
        """
        for k in self._model_list_dict.keys():
            self.write_data(k)

    def write_data(self, entity: str) -> None:
        """
        Write cached entity/relation data to the corresponding CSV file.

        Args:
            entity (str): The name of the entity/relation to write.

        Returns:
            None
        """
        if len(self._model_list_dict[entity]) > 0:
            start = time.time()
            path = self._csv_paths[entity]
            data = pd.read_csv(path).set_index("index")
            cp1 = time.time()
            new_data = pd.DataFrame(
                [x.model_dump() for x in self._model_list_dict[entity]]
            )
            new_data.index.name = "index"
            updated_df = pd.merge(
                data, new_data, on="id", how="outer", suffixes=("", "_new")
            )
            # Replace old values with new ones where new values are not null
            for column in new_data.columns:
                if column != "id":
                    updated_df[column] = updated_df[column].combine_first(
                        updated_df[column + "_new"]
                    )
                    updated_df = updated_df.drop(columns=[column + "_new"])
            updated_df = updated_df.drop_duplicates(subset="id")
            updated_df.reset_index()
            updated_df.index.name = "index"
            updated_df.to_csv(path, index_label="index")
            cp2 = time.time()
            self.empty_entity_dict(entity)
            read_old = str(cp1 - start)
            write_new = str(cp2 - cp1)
            logging.info(f"reading existing data took {read_old}")
            logging.info(f"reading and writing new data took {write_new}")

    @property
    def incomplete_relations(self):
        """
        Retrieve all incomplete relations from the CSV files.

        Args:
            None

        Returns:
            Dict[str, List[List[str]]]: A dictionary where each key is a relation name and the value is a list of incomplete relation entities.
        """
        relations = self._schema.relations
        incomplete_res = {}

        for k in relations.keys():
            incomplete_res[k] = []
            path = self._csv_paths[k]
            df = pd.read_csv(path).set_index("index")
            filtered_df = df.loc[df["completed"] == False]
            for index, row in filtered_df.iterrows():
                row = row.map(lambda x: "None" if pd.isna(x) else x)
                relation_list = [row["name"], row["source"], row["target"]]
                incomplete_res[k].append(relation_list)
        return incomplete_res

    def set_relation_complete(self, relation):
        """
        Mark a specific relation as complete in its CSV file.

        Args:
            relation (str): The name of the relation to mark as complete.

        Returns:
            None
        """
        path = self._csv_paths[relation]
        df = pd.read_csv(path).set_index("index")
        df["completed"] = True
        df.index.name = "index"
        df.to_csv(path)

    def print_data(self):
        """
        Print the count of data entries for each entity and relation.

        Args:
            None

        Returns:
            None
        """
        for k, v in self._model_data.items():
            print(f"{k}: {v}.")

    @property
    def nodes_dict(self) -> dict:
        """
        Retrieve all nodes (entities) as a dictionary of DataFrames.

        Args:
            None

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where each key is an entity name and the value is its corresponding DataFrame.
        """
        entities = self.schema.entities
        nodes = {}
        for e in entities.keys():
            path = self._csv_paths[e]
            df = pd.read_csv(path).set_index("index")
            df_copy = df.copy(deep=True)
            nodes[e] = df_copy
        return nodes

    @property
    def relation_dict(self) -> dict:
        """
        Retrieve all relations as a dictionary of DataFrames.

        Args:
            None

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where each key is a relation name and the value is its corresponding DataFrame.
        """
        relations = self.schema.relations
        rels = {}
        for r in relations.keys():
            path = self._csv_paths[r]
            df = pd.read_csv(path).set_index("index")
            df_copy = df.copy(deep=True)
            rels[r] = df_copy
        return rels

    def remove_duplicates(self):
        """
        Remove duplicates by re-computing ids based on name only.

        IMPORTANT:
            After this function is called, the csv files cannot be used to generate anymore,
            so this function should only be called at the very end of the generation.

        """

        def generate_id(row: pd.Series) -> str:
            name = row[columns[1]]
            if not isinstance(name, str):
                name = str(name)
            # Generate a UUID
            return str(hashlib.sha256(name.encode("utf-8")).hexdigest())

        # remove entities with same names
        entities = self.schema.entities
        for e in entities.keys():
            path = self._csv_paths[e]
            df = pd.read_csv(path).set_index("index")
            columns = df.columns[1:-1]
            df["id"] = df.apply(generate_id, axis=1)
            # compute the id based on name
            df = df.drop_duplicates(["id"], ignore_index=True)
            print(f"{e}: {len(df)}")
            path = self._csv_paths[e]
            df.to_csv(path, index_label="index")

        # remove relations if source or target do not appear
        relations = self.schema.relations
        for rel in relations.keys():
            path = self._csv_paths[rel]
            df = pd.read_csv(path).set_index("index")
            print(f"Relation: {rel}")
            e1, e2 = self.schema.find_relation(rel)
            path1 = self._csv_paths[e1]
            df_e1 = pd.read_csv(path1).set_index("index")
            path2 = self._csv_paths[e2]
            df_e2 = pd.read_csv(path2).set_index("index")
            df = df[df["source"].isin(df_e1[df_e1.columns[2]])]
            df = df[df["target"].isin(df_e2[df_e2.columns[2]])]
            df = df.drop_duplicates(["name", "source", "target"], ignore_index=True)
            path = self._csv_paths[rel]
            df.to_csv(path, index_label="index")
        logging.info("Duplicated entities and relations are removed.")

    def print_data_stats(self):
        total_entities = 0
        for entity in self.nodes_dict:
            print(f"{entity}: {len(self.nodes_dict[entity])}")
            total_entities += len(self.nodes_dict[entity])
        print(f"Total entities: {total_entities}")
        total_relations = 0
        for rel in self.relation_dict:
            print(f"{rel}: {len(self.relation_dict[rel])}")
            total_relations += len(self.relation_dict[rel])
        print(f"Total relations: {total_relations}")

    def post_process_csv_files(self) -> None:
        """
        This function turns the final csv files into the desired form.

        IMPORTANT:
            After this function is called, the csv files cannot be used to generate anymore,
            so this function should only be called at the very end of the generation.
        """

        def map_name_to_id(name, mapping, entity_type):
            if name in mapping:
                return mapping[name]
            else:
                print(
                    f"Warning: {entity_type} name '{name}' not found in entities. Setting as 'None'."
                )
                return None

        for source_en, target_en, rel in self.schema.relation_triples:
            source_path = self._csv_paths[source_en]
            source_primary = list(
                self._schema.get_attribute_for_entity(source_en).keys()
            )[0]
            target_path = self._csv_paths[target_en]
            target_primary = list(
                self._schema.get_attribute_for_entity(target_en).keys()
            )[0]
            rel_path = self._csv_paths[rel]

            edges = pd.read_csv(rel_path).set_index("index")
            source_vertices = pd.read_csv(source_path).set_index("index")
            target_vertices = pd.read_csv(target_path).set_index("index")

            edges = edges.rename(
                columns={
                    "id": "hash",
                    "source": "source_name_id",
                    "target": "target_name_id",
                }
            )

            source_name_to_id = pd.Series(
                source_vertices.id.values, index=source_vertices[source_primary]
            ).to_dict()
            target_name_to_id = pd.Series(
                target_vertices.id.values, index=target_vertices[target_primary]
            ).to_dict()

            print(source_name_to_id)
            print(target_name_to_id)

            edges["source_name_id"] = edges["source_name_id"].apply(
                lambda x: map_name_to_id(x, source_name_to_id, "Source")
            )
            edges["target_name_id"] = edges["target_name_id"].apply(
                lambda x: map_name_to_id(x, target_name_to_id, "Target")
            )

            # Optionally, you can drop rows where mapping was not found
            missing_sources = edges["source_name_id"].isnull().sum()
            missing_targets = edges["target_name_id"].isnull().sum()

            if missing_sources > 0 or missing_targets > 0:
                print(
                    f"Info: Dropping {missing_sources + missing_targets} relations due to missing entity mappings."
                )
                edges = edges.dropna(subset=["source_name_id", "target_name_id"])

            processed_relations = edges.reset_index()
            processed_relations.to_csv(rel_path, index=False)
