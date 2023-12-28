# Running lm-evaluation-harness Tests via Docker

This guide outlines the steps to install and run tests for lm-evaluation-harness using Docker.

## Step 0: Requirements
- **Python**
- **Docker**

## Step 1: Pull Docker Image

```bash
docker pull daniilbarinov/lm-eval:1.0
```

## Step 2: Verify Image Download

```bash
docker images
```

Ensure that the image *daniilbarinov/lm-eval:1.0* is listed.

## Step 3: Obtain 'run_docker_tests.py'

```bash
git clone https://github.com/Deelvin/lm-evaluation-harness.git -b dbarinov/unittest_endpoints_docker
cd lm-evaluation-harness/tests
```
or:
```bash
curl -O https://raw.githubusercontent.com/Deelvin/lm-evaluation-harness/dbarinov/unittest_endpoints_docker/run_docker_tests.py
```

## Step 4: Set Token Environment Variable
```bash
export OCTOAI_TOKEN=<your_token>
```
Note: currently endpoints unittests are working only for **'Prod'** token.

## Step 5: Run Tests via Docker
```bash
python run_docker_tests.py
```

This script will start a Docker container with tests and save the results to the *test_results/test_results.csv* file.