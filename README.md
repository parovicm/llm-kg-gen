# Generating Domain-Specific Knowledge Graphs from Large Language Models

This repository contains code for the
paper: [Generating Domain-Specific Knowledge Graphs from Large Language Models](https://aclanthology.org/2025.findings-acl.602/).
We propose a prompt-based method to automatically construct
domain-specific knowledge graphs (KGs) by extracting knowledge directly from large language models' (LLMs) parameters.
First, we use an LLM to construct a schema which contains a set of domain-representative entities and relations. This
schema is then used to guide the LLM through an iterative data generation process equipped
with [Chain-of-Verification](https://aclanthology.org/2024.findings-acl.212/)
for increased data quality.
We then propose to evaluate generated KGs against [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page).

## Environment Setup

Set up a conda environment:

```
conda create --name kg_generation python=3.11
conda activate kg_generation
pip install -r requirements.txt
```

## Schema Generation

To generate schema for a domain of interest use `kg_construction/schema_generation/schema_generation.py`. We
recommend `two_step_generation` where the first step generates entities and relations, and the second step adds
attributes to entities. The schema is then post-processed to ensure it follows the correct format.

Schema generation is an important step of our method because schema quality will determine the content and quality of
the
generated KGs. Therefore, we suggest manually checking and modifying generated schema to ensure that 1)
its content meets your expectations, and 2) it can successfully be used for LLM data generation.

## Data Generation

Data generation produces CSV files that contain KG entities and relations. There are three generation steps (implemented
in `kg_construction/data_generation/data_generation_with_csv.py`):

1. **Initial Relation Generation**: for all relations, generate `N` pairs of entities,
2. **Entity Completion (EnComp)**: given a pair of entities, complete all remaining entities and attributes in the
   schema,
3. **Entity Generation (EnGen)**: given a pair of entities, generate `M` more main entities. Data generated in this step
   is validated
   using `Chain of Verification` method before being used
   for further generation. The implementation for `Chain of Verification` is based on
   this [repository](https://github.com/ritun16/chain-of-verification).

To run data generation, specify the schema, desired parameters, and LLM in `main.py`, and start the generation
process.

## Wikidata Extraction and Evaluation

Download `Wikidata` dump and process it following the
instructions [here](https://github.com/neelguha/simple-wikidata-db). We store this code in `simple_wikidata_db`.

After that, extract the domain-specific data (`wikidata/extraction`):

```
python run_extraction.py --config book_config.yaml
```

Finally, evaluate generated KGs against domain-specific KGs extracted from `Wikidata` (`wikidata/evaluation`):

```
python run_evaluation.py --config book_config.yaml
```

## Citation

If you use this code, please cite the following paper:

```
@inproceedings{parovic-etal-2025-generating,
    title = "Generating Domain-Specific Knowledge Graphs from Large Language Models",
    author = "Parovi{\'c}, Marinela  and
      Li, Ze  and
      Du, Jinhua",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.602/",
    pages = "11558--11574",
    ISBN = "979-8-89176-256-5",
    abstract = "Knowledge graphs (KGs) have been a cornerstone of search and recommendation due to their ability to store factual knowledge about any domain in a structured form enabling easy search and retrieval. Large language models (LLMs) have shown impressive world knowledge across different benchmarks and domains but their knowledge is inconveniently scattered across their billions of parameters. In this paper, we propose a prompt-based method to construct domain-specific KGs by extracting knowledge solely from LLMs' parameters. First, we use an LLM to create a schema for a specific domain, which contains a set of domain-representative entities and relations. After that, we use the schema to guide the LLM through an iterative data generation process equipped with Chain-of-Verification (CoVe) for increased data quality. Using this method, we construct KGs for two domains: books and landmarks, which we then evaluate against Wikidata, an open-source human-created KG. Our results show that LLMs can generate large domain-specific KGs containing tens of thousands of entities and relations. However, due to the increased hallucination rates as the procedure evolves, the utility of large-scale LLM-generated KGs in practical applications could remain limited."
}
```