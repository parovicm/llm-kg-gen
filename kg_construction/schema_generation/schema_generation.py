import argparse
import json
import os
import re
import sys
from datetime import datetime
from string import Template

sys.path.append("/ccs_data/marinela/projects/llm-kg-gen")

from kg_construction.utils.sync_llm_utils import LLM
from schema_post_processing import post_processing

# Base directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

prompt_file1 = os.path.join(base_dir, "../../prompts/schema/one_step.json")
prompt_file2 = os.path.join(base_dir, "../../prompts/schema/two_step_entities.json")
prompt_file3 = os.path.join(base_dir, "../../prompts/schema/two_step_attributes.json")

schema_shot_file = Template(os.path.join(base_dir, "../../schema/shots/${topic}.json"))
entity_shot_file = Template(
    os.path.join(base_dir, "../../schema/shots/${topic}_entities.json")
)

res_path = Template(
    os.path.join(
        base_dir,
        "../../schema/${domain}/step_${step}_topic_${topic}_temp_${temp}_model_${model}.json",
    )
)


def parse_and_save_json(text, output):
    json_pattern = r"\{.*\}"
    matches = re.findall(json_pattern, text, re.DOTALL)
    if len(matches) != 1:
        return None
    content = None
    match = matches[0]
    try:
        content = json.loads(match)
    except json.decoder.JSONDecodeError as e:
        pass
    if content is None:
        # use regular expressions to try to fix the invalid JSON format
        match = match.replace('"', "'")
        match = re.sub(r"(.*[\"']?,?)\s+#.*", r"\1", match)
        match = re.sub(r"(([{:]\s*)|(}?,\s*))'", r'\1"', match, flags=re.DOTALL)
        match = re.sub(
            r"'((:\s*)|(:\s*{)|(,\s*\n)|(\s*}))", r'"\1', match, flags=re.DOTALL
        )
        match = re.sub(r"\\'", "'", match, flags=re.DOTALL)
        match = match.replace("None", "null").replace('"None"', "null")
        try:
            content = json.loads(match)
        except json.decoder.JSONDecodeError as e:
            pass
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if content:
        with open(output, "w+", encoding="utf-8") as f:
            json.dump(content, ensure_ascii=False, indent=4, fp=f)
            print(f"JSON extracted and saved to: {output}")
    else:
        print("No valid JSON found in the provided text.")
    return content


def construct_prompt(
        domain: str,
        prompt_file: str,
        shot_schema: Template,
        shot_topic: str,
        domain_schema: str = None,
) -> dict:
    config = {"topic": shot_topic, "domain": domain}
    shot_schema = shot_schema.substitute(topic=shot_topic)
    with open(shot_schema, "r", encoding="utf-8") as schema_data:
        schema = json.load(schema_data)
        config["topic_schema"] = schema

    if domain_schema:
        with open(domain_schema, "r", encoding="utf-8") as domain_schema_data:
            schema = json.load(domain_schema_data)
            config["domain_schema"] = schema

    with open(prompt_file, "r", encoding="utf-8") as prompt_data:
        prompt = json.load(prompt_data)
        prompt = {k: v.format(**config) for k, v in prompt.items()}
    return prompt


def generate_response(
        prompt: dict,
        domain: str,
        topic: str,
        output_path: Template,
        temperature: float = 0.2,
        model_name: str = "llama3.1-70b",
        marker: int = 0,
) -> (str, int):
    start_time = datetime.now()

    system_prompt = prompt["system_message"]
    user_prompt = prompt["user_message"]

    marker_method = {0: "one", 1: "entities", 2: "attributes"}
    llm = LLM(temperature=temperature)

    path = output_path.substitute(
        domain=domain,
        temp=llm.temp().replace(".", "_"),
        step=marker_method[marker],
        topic=topic,
        model=model_name.replace(".", "_"),
    )
    print("Output: " + path)
    res, _ = llm.call(
        user_prompt, system_prompt, model_name=model_name, temperature=temperature
    )
    parse_and_save_json(res, path)

    end_time = datetime.now()
    time = end_time - start_time
    return path, time


def one_step_generation(
        domain: str,
        prompt: str = prompt_file1,
        shot_schema: Template = schema_shot_file,
        shot_topic: str = "video",
        output_template: Template = res_path,
        temperature: float = 0.2,
        model_name: str = "llama3.1-70b",
) -> (str, int):
    prompt = construct_prompt(
        domain=domain,
        prompt_file=prompt,
        shot_schema=shot_schema,
        shot_topic=shot_topic,
    )
    print(prompt)
    path, time = generate_response(
        prompt=prompt,
        domain=domain,
        topic=shot_topic,
        output_path=output_template,
        temperature=temperature,
        model_name=model_name,
        marker=0,
    )
    return path, time


def two_step_generation(
        domain: str,
        prompt_entity: str = prompt_file2,
        prompt_attribute: str = prompt_file3,
        shot_schema_entity: Template = entity_shot_file,
        shot_schema_attribute: Template = schema_shot_file,
        shot_topic: str = "video",
        output_template: Template = res_path,
        temperature: float = 0.2,
        model_name: str = "llama3.1-70b",
) -> (str, int):
    # Generate entities and relations
    prompt = construct_prompt(
        domain=domain,
        prompt_file=prompt_entity,
        shot_schema=shot_schema_entity,
        shot_topic=shot_topic,
    )
    print(prompt)
    path1, time1 = generate_response(
        prompt=prompt,
        domain=domain,
        topic=shot_topic,
        output_path=output_template,
        temperature=temperature,
        model_name=model_name,
        marker=1,
    )
    # Generate attributes
    prompt = construct_prompt(
        domain=domain,
        prompt_file=prompt_attribute,
        shot_schema=shot_schema_attribute,
        shot_topic=shot_topic,
        domain_schema=path1,
    )
    print(prompt)
    path2, time2 = generate_response(
        prompt=prompt,
        domain=domain,
        topic=shot_topic,
        output_path=output_template,
        temperature=temperature,
        model_name=model_name,
        marker=2,
    )
    return path2, time1 + time2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="Domain of knowledge graph.")
    args = parser.parse_args()
    domain = args.domain
    shot_topic = "video"
    # path, _ = one_step_generation(domain=domain, shot_topic=shot_topic, model_name="llama3.1-70b")
    path, _ = two_step_generation(
        domain=domain, shot_topic=shot_topic, model_name="llama3.1-70b"
    )
    post_processing([path])
