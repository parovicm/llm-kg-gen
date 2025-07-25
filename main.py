import asyncio
import os

from kg_construction.data_generation.data_generation_with_csv import (
    async_generate_initial_relations,
    async_complete_entities,
    async_generate_entities,
)
from kg_construction.data_generation.data_manager.csv_manager import CsvManager
from kg_construction.data_generation.data_manager.schema_interface import SchemaEnglish

if __name__ == "__main__":
    print(os.getcwd())
    base_dir = os.path.dirname(os.path.abspath(__file__))
    schema_file = os.path.join(
        base_dir, "schema/books/book_schema_processed_categories.json"
    )
    # schema_file = os.path.join(base_dir, "schema/landmarks/landmarks_schema_processed_categories.json")
    schema_obj = SchemaEnglish(schema_file)
    output = os.path.join(base_dir, "test_output")
    domain = "books_test"
    manager = CsvManager(schema_obj, output, domain)

    # Step 1: Initial Relation Generation
    prompt_file = "./prompts/generation/generate_initial_relations.json"
    asyncio.run(
        async_generate_initial_relations(prompt_file, 20, manager, model="llama3.1-70b")
    )

    # Step 2: Entity Completion (EnComp): complete remaining entities and relations based on given pairs
    prompt_file = "./prompts/generation/complete_missing_entities_tags.json"
    asyncio.run(async_complete_entities(prompt_file, manager, model="llama3.1-70b"))

    for _ in range(3):
        # Step 3: Entity Generation (EnGen): generate more main entity identifiers
        prompt_file = "./prompts/generation/generate_more_entities.json"
        asyncio.run(
            async_generate_entities(
                prompt_file, manager, 10, model="llama3.1-70b", use_cove=True
            )
        )
        # Step 2: Entity Completion (EnComp)
        prompt_file = "./prompts/generation/chinese/complete_missing_entities_tags.json"
        asyncio.run(async_complete_entities(prompt_file, manager))

    # Remove multiple instances of entities with the same name.
    # manager.remove_duplicates()
    # Post process CSV files to obtain entity ids (in addition to names) in relation files
    # manager.post_process_csv_files()
