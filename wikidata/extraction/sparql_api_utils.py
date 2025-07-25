import os
import time

import pandas as pd
from typing import List, Dict
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib


class WikiDataQueryResults:
    """
    A class that can be used to query data from Wikidata using SPARQL and return the results as a Pandas DataFrame or a list
    of values for a specific key.
    """

    def __init__(self, output_dir=""):
        """
        Initializes the WikiDataQueryResults object with a SPARQL query string.
        :param query: A SPARQL query string.
        """
        self.endpoint_url = "https://query.wikidata.org/sparql"
        self.sparql = SPARQLWrapper(self.endpoint_url)
        self.sparql.addCustomHttpHeader("User-Agent", "Chrome/126.0.6478.127")
        self.output_dir = output_dir

    def run_query(self, query):
        """
        Set and execute the given query.
        Args:
            query: query to be executed.

        Returns:
        Result of the query.
        """
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        return self._load()

    @staticmethod
    def __transform2dicts(results: List[Dict]) -> List[Dict]:
        """
        Helper function to transform SPARQL query results into a list of dictionaries.
        :param results: A list of query results returned by SPARQLWrapper.
        :return: A list of dictionaries, where each dictionary represents a result row and has keys corresponding to the
        variables in the SPARQL SELECT clause.
        """
        new_results = []
        for result in results:
            new_result = {}
            for key in result:
                new_result[key] = result[key]["value"]
            new_results.append(new_result)
        return new_results

    def _load(self) -> List[Dict]:
        """
        Helper function that loads the data from Wikidata using the SPARQLWrapper library, and transforms the results into
        a list of dictionaries.
        :return: A list of dictionaries, where each dictionary represents a result row and has keys corresponding to the
        variables in the SPARQL SELECT clause.
        """
        retries = 5
        results = None
        for _ in range(retries):
            try:
                results = self.sparql.queryAndConvert()["results"]["bindings"]
                break
            except urllib.error.HTTPError as e:
                print(f"Error: {e}")
                time.sleep(5)
        results = self.__transform2dicts(results)
        return results

    def _get_filename(self, item_id: str, relation_id: str = None):
        if relation_id:
            os.path.join(self.output_dir, f"{item_id}_{relation_id}.csv")
        return os.path.join(self.output_dir, f"{item_id}.csv")

    def load_as_dataframe(self, item_id: str, relation_id: str = None) -> List[Dict]:
        """
        Executes the SPARQL query and returns the results as a Pandas DataFrame.
        :return: A Pandas DataFrame representing the query results.
        """
        filename = self._get_filename(item_id, relation_id)
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df = df.drop(["index"], axis=1)
            return df.to_dict(orient="records")
        else:
            return []

    def save_as_dataframe(
        self, item_id, results, relation_id: str = None
    ) -> pd.DataFrame:
        """
        Saves results as a dataframe to a correct file.
        Args:
            item_id:
            results: results dataframe
            relation_id: optional relation, otherwise only entity

        Returns:
            None
        """
        filename = self._get_filename(item_id, relation_id)
        if len(results) == 0:
            if relation_id is not None:
                df = pd.DataFrame(
                    columns=["item", "itemLabel", "relation", "relationLabel"]
                )
            else:
                df = pd.DataFrame(columns=["item", "itemLabel"])
                # df = pd.DataFrame(columns=["alias"])
        else:
            df = pd.DataFrame.from_dict(results)
        df.to_csv(filename, index=False)
        return df

    def get_values_for_item(self, item_id: str) -> List:
        """
        Get values for a given item.
        Args:
            item_id: Wikidata id

        Returns:
        Results from Wikidata query.
        """
        query = f"""
                SELECT ?item ?itemLabel
                    WHERE   {{
                        ?item wdt:P31 wd:{item_id};
                        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
                }}
        """
        output_file = self._get_filename(item_id)
        if os.path.exists(output_file):
            # load file and return
            results = self.load_as_dataframe(item_id)
        else:
            print(f"Item: {item_id}")
            results = self.run_query(query)
            self.save_as_dataframe(item_id, results)
        return results

    def get_alias(self, item_id: str) -> List:
        """
        Get aliases for a given item.
        Args:
            item_id:

        Returns:
            A dataframe with a single column containing aliases.
        """
        query = f"""
                SELECT ?alias 
                    WHERE   {{
                        wd:{item_id} skos:altLabel ?alias.
                FILTER (LANG(?alias) = "en")
                }}
        """
        output_file = self._get_filename(item_id)
        if os.path.exists(output_file):
            # load file and return
            results = self.load_as_dataframe(item_id)
        else:
            results = self.run_query(query)
        return results

    def get_relation_values_for_item(self, item_id: str, relation_id: str) -> List:
        query = f"""
                SELECT DISTINCT ?item ?itemLabel ?relation ?relationLabel
                    WHERE {{
                      ?item wdt:P31 wd:{item_id};
                        wdt:{relation_id} ?relation.
                    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
                    }}
                """
        output_file = self._get_filename(item_id, relation_id)
        if os.path.exists(output_file):
            # load file and return
            results = self.load_as_dataframe(item_id, relation_id)
        else:
            results = self.run_query(query)
            self.save_as_dataframe(item_id, results, relation_id)
        return results


def get_direct_subclasses_for_item(item_id: str, output_dir: str):
    """
    Get direct subclasses (property wdt:P279) for a given item.
    Args:
        item_id: Wikidata id

    Returns:
    List of ids for direct subclasses of item.
    """
    query = f"""SELECT ?item ?itemLabel 
                    WHERE {{
                        ?item wdt:P279 wd:{item_id}.
                    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
                    }}
        """
    results = WikiDataQueryResults().run_query(query)
    if len(results) == 0:
        df = pd.DataFrame(columns=["item", "itemLabel"])
    else:
        df = pd.DataFrame.from_dict(results)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"{item_id}.csv"), index=False)
    return results


def get_all_subclasses_for_item(item_id: str, output_dir: str) -> List[Dict]:
    """
    Get values for a given item.
    Args:
        item_id: Wikidata id
        output_dir: directory to output the result

    Returns:
    Results from Wikidata query.
    """
    query = f"""SELECT ?item ?itemLabel
                    WHERE   {{
                        ?item wdt:P279* wd:{item_id};
                        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                    }}
        """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{item_id}.csv")
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        return df.to_dict(orient="records")
    else:
        results = WikiDataQueryResults().run_query(query)
        df = pd.DataFrame.from_dict(results)
        df.to_csv(filename, index=False)
        return results


def get_values_for_item_and_all_subclasses(item_id: str):
    """
    Get values for a given item. This fails when the result is too large.
    Args:
        item_id: Wikidata id

    Returns:
    Results from Wikidata query.
    """
    query = f"""
                SELECT ?item ?itemLabel
                    WHERE   {{
                        ?item wdt:P31/wdt:P279* wd:{item_id};
                        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                    }}
        """
    return WikiDataQueryResults().run_query(query)
