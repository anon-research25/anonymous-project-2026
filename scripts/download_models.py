"""Download requested HF assets and write their actual resolved snapshot identities."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from ammfv.config import load_config
from ammfv.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datasets/mocheg.yaml")
    parser.add_argument("--components", nargs="+", choices=["clip", "bge", "llm", "router_base"],
                        default=["clip", "bge", "llm", "router_base"])
    parser.add_argument("--manifest", default="outputs/pretrained_assets.json")
    args = parser.parse_args()
    config = load_config(args.config)
    components = {"clip": config["encoders"]["clip"], "bge": config["encoders"]["bge"],
                  "llm": config["llm"], "router_base": config["router_training"]}
    records = {}
    for name in args.components:
        settings = components[name]
        snapshot = snapshot_download(repo_id=settings["model"], revision=settings["revision"])
        records[name] = {"model": settings["model"], "requested_revision": settings["revision"],
                         "resolved_snapshot": snapshot, "commit": Path(snapshot).name}
    write_json(args.manifest, records)


if __name__ == "__main__":
    main()
