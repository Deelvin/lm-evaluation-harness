Adding test to CI
===============


#### Run tests locally

At the moment, the tests use an already deployed endpoint. To run tests locally, you need to add the endpoint URL to your environment variables. Also for Octoai endpoints you need to add the Octoai token to the environment variable.

``` bash
    export ENDPOINT="url to endpoint"
    export OCTOAI_TOKEN="octoai token"
```

To run tests use the following command:
``` bash
    pytest tests/smoke_tests.py --model_name llama-2-7b-chat
```

#### Run tests CI


In order to run tests in CI, you need to modify `build.yaml`. To do this, you need to add a new task and register new steps. You also need to add environment variables such as `ENDPOINT` and `OCTOAI_TOKEN`. 
So the new task should look like this:

``` yaml
    smoke_test:
    name: Smoke test
    runs-on: ubuntu-latest
    needs: temp_deploy_for_downstream_actions
    steps:
      - name: Test endpoint
        uses: actions/checkout@v3

      - name: Set up Python 
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install pytest pytest-cov
          pip install openai
          pip install -U sentence-transformers
      - env:
          ENDPOINT: ${{ needs.temp_deploy_for_downstream_actions.outputs.endpoint }}
          OCTOAI_TOKEN: ${{ secrets.OCTOAI_TOKEN }}
        run: |
          echo "Run smoke testing on $ENDPOINT"
          pytest tests/smoke_tests.py --model_name llama-2-7b-chat

```

To test the work of a new task, you can create a PR and a new task should appear there, or use the following project to run github actions locally: [link](https://github.com/nektos/act)
