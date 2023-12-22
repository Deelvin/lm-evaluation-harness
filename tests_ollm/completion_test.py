import argparse
import requests
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="codellama-7b-fp16",
        help="Model name",
    )
    parser.add_argument(
        "-ip",
        "--ip",
        type=str,
        default="0.0.0.0",
        help="IP address of ollm server",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=32777,
        help="port of ollm server",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=None,
        help="Temperature parameter",
    )
    parser.add_argument(
        "-tp",
        "--top_p",
        type=float,
        default=None,
        help="Top_p parameter",
    )

    return parser.parse_args()


def create_payload(args):
    payload = {
        "model": args.model,
        "prompt": "What is the capital of France?",
    }

    if args.top_p is not None:
        payload["top_p"] = args.top_p

    if args.temperature is not None:
        payload["temperature"] = args.temperature

    return payload


def main():
    args = parse_args()

    token = os.environ["OCTOAI_TOKEN"]
    headers = {
        "authorization": f"Bearer {token}",
        "content-type": "application/json",
    }

    payload = create_payload(args)
    response = requests.post(
        f"http://{args.ip}:{args.port}/v1/completions", json=payload, headers=headers
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")

    response = json.loads(response.text)
    answer = response["choices"][0]["text"]
    print("Answer:", answer)


if __name__ == "__main__":
    main()
