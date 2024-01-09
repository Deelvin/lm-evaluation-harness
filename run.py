import subprocess
import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type", type=str, default="benchmark",
    )
    args = parser.parse_args()

    if args.type == "benchmark":
        subprocess.run(
            "python scripts/run_endpoints_benchmark.py",
            shell=True,
            universal_newlines=True
        )
    elif args.type == "unittests":
        subprocess.run(
            "python tests/run_docker_tests.py",
            shell=True,
            universal_newlines=True
        )
    else:
        raise RuntimeError("Unknown run type specified: choose 'benchmark' or 'unittests'")

if __name__ == "__main__":
    main()