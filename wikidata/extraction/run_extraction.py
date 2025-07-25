import argparse

import yaml

from utils import extract_wikidata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    extract_wikidata(**config)


if __name__ == "__main__":
    main()
