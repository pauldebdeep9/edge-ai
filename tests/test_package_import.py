import edge_ai


def test_package_can_be_imported() -> None:
    """The source-layout package is available to the test environment."""
    assert edge_ai.__name__ == "edge_ai"
