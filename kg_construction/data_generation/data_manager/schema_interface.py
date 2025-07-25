import abc
import json
import os
from abc import ABCMeta
from collections import defaultdict

from kg_construction.utils.format_enforcing_utils import (
    create_pydantic_classes_from_schema,
)


class Schema(metaclass=ABCMeta):

    @property
    @abc.abstractmethod
    def path(self):
        pass

    @property
    @abc.abstractmethod
    def file_name(self):
        pass

    @property
    @abc.abstractmethod
    def schema(self):
        pass

    @property
    @abc.abstractmethod
    def relationless_schema(self):
        pass

    @property
    @abc.abstractmethod
    def entities(self):
        pass

    @property
    @abc.abstractmethod
    def main_entity(self):
        pass

    @property
    @abc.abstractmethod
    def relations(self):
        pass

    @property
    @abc.abstractmethod
    def relation_triples(self):
        pass

    @property
    @abc.abstractmethod
    def pydantic_class(self):
        pass

    @abc.abstractmethod
    def validate(self):
        pass

    @abc.abstractmethod
    def find_relation_between_entities(self, entity1: str, entity2: str):
        pass

    @abc.abstractmethod
    def find_relation(self, relation_name: str):
        pass

    @abc.abstractmethod
    def get_attribute_for_entity(self, entity):
        pass

    @abc.abstractmethod
    def add_attribute_to_entity(self, entity, attribute, value_type):
        pass

    @abc.abstractmethod
    def update_schema(self):
        pass


class SchemaEnglish(Schema):
    def __init__(self, path):
        self._path = path
        with open(path, "r", encoding="utf-8") as f:
            self._schema = json.load(f)

    @property
    def path(self):
        return self._path

    @property
    def file_name(self):
        _, schema_file = os.path.split(self.path)
        schema_file, _ = os.path.splitext(schema_file)
        schema_file = os.path.splitext(schema_file)
        return str(schema_file)

    @property
    def schema(self):
        return self._schema

    @property
    def relationless_schema(self):
        result = {}
        for entity in self.schema["concepts"]:
            result[entity] = self.schema["concepts"][entity]["attributes"]
        return result

    def get_attribute_for_entity(self, entity) -> dict:
        return self.entities[entity]["attributes"]

    @property
    def entities(self):
        return self.schema["concepts"]

    @property
    def relations(self):
        return self.schema["relations"]

    @property
    def relation_triples(self):
        """
        Extract triples (entity1, entity2, relation_name) from the schema. It assumes schema is in the correct format.

        Args:

        Returns:
            list: list of (entity1, entity2, relation_name) triples from the schema
        """
        relations = self.schema["relations"]
        relations_list = []
        for rel in relations:
            rel_name = rel
            entity1 = relations[rel]["source"]
            entity2 = relations[rel]["target"]
            relations_list.append((entity1, entity2, rel_name))
        return relations_list

    @property
    def pydantic_class(self):
        return create_pydantic_classes_from_schema(self.schema, class_name="schema")

    @property
    def main_entity(self):
        """
        The main entity is the one appearing the most times in the relations
        Returns:
            the main entity in the schema
        """
        relations = self.schema["relations"]
        entity_freq = defaultdict(int)
        for rel in relations:
            entity1 = relations[rel]["source"]
            entity2 = relations[rel]["target"]
            entity_freq[entity1] += 1
            entity_freq[entity2] += 1
        return max(entity_freq, key=lambda x: entity_freq[x])

    def validate(self):
        return

    def find_relation_between_entities(self, entity1: str, entity2: str):
        """
        Find the relation between 2 entities given its key name.
        """
        all_relations = self.relation_triples
        for en1, en2, rel in all_relations:
            if {en1, en2} == {entity1, entity2}:
                return en1, en2, rel
        return None

    def find_relation(self, relation_name: str):
        """
        Find a specific relation in a list of (entity1, entity2, relation) triples.
        Args:
            relation_name: Relation to find

        Returns:
            A pair of entities connected by the provided relation
        """
        all_relations = self.relation_triples
        for entity1, entity2, rel in all_relations:
            if rel == relation_name:
                return entity1, entity2
        return "", ""

    def add_attribute_to_entity(self, entity, attribute, value_type):
        if entity in self.entities:
            self.entities[entity]["attributes"][attribute] = value_type
            self.update_schema()
        else:
            pass

    def update_schema(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.schema, ensure_ascii=False, indent=4, fp=f)
