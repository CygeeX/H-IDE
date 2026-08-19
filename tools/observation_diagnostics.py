from __future__ import annotations

import argparse
import json

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8092")
    parser.add_argument("--token", default="dev-control-token")
    args = parser.parse_args()
    response = httpx.get(
        f"{args.base_url}/internal/v1/capabilities",
        headers={"Authorization": f"Bearer {args.token}"},
        timeout=5,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

