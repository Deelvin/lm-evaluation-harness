def pytest_addoption(parser):
    parser.addoption("--model_name", default=None, type=str, action="store")
    parser.addoption("--endpoint", default=None, type=str, action="store")
    parser.addoption("--context_size", type=int, default=4096, action="store")
