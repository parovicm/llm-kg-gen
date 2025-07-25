import argparse
import asyncio
import os
import sys
from pprint import pprint

from dotenv import load_dotenv

from cove_chains import EntityListCOVEChain
from custom_llm import CustomOpenAI

sys.path.append(os.path.join(os.getcwd(), "kg_construction", "cove"))
load_dotenv("/workspace/.env")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chain of Verification (CoVE) parser.")
    parser.add_argument(
        "--question",
        type=str,
        required=False,
        default='Generate 10 books with author Charles Dickens. If there are no matching books output "None". \n1. Oliver Twist\n',
        help="The original question user wants to ask",
    )
    parser.add_argument(
        "--llm-name",
        type=str,
        required=False,
        default="llama3.1-70b",
        help="The llm name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=0.1,
        help="Temperature of the llm",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=False,
        default=2000,
        help="max_tokens of the llm",
    )
    parser.add_argument(
        "--show-intermediate-steps",
        type=bool,
        required=False,
        default=True,
        help="Show intermediate steps from different chains",
    )
    args = parser.parse_args()

    original_query = args.question
    chain_llm = CustomOpenAI(
        model_name=args.llm_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    cove_chain_instance = EntityListCOVEChain(chain_llm, baseline_response_exists=False)
    cove_chain = cove_chain_instance()
    cove_chain_result = asyncio.run(
        cove_chain.ainvoke({"original_question": original_query})
    )

    if args.show_intermediate_steps:
        print("\n" + 80 * "#" + "\n")
        pprint(cove_chain_result)
        print("\n" + 80 * "#" + "\n")
    print("Final Answer: {}".format(cove_chain_result["final_answer"]))
