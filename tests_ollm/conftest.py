def pytest_addoption(parser):
    parser.addoption("--model_name", required=True, type=str, action="store")
    parser.addoption("--endpoint", required=True, type=str, action="store")
    parser.addoption("--context_size", type=int, default=2048, action="store")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.model_name
    if "model_name" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("model_name", [option_value])
