"""The window shim's argument parsing must be safe against malformed input --
the shim usually runs under pyw, where nothing printed is ever seen, so a
bad --url must degrade to the default instead of navigating to garbage."""

from enigma_window import DEFAULT_URL, _host_port, _parse_args


def test_defaults():
    assert _parse_args([]) == (DEFAULT_URL, False)


def test_on_top_flag():
    assert _parse_args(["--on-top"]) == (DEFAULT_URL, True)


def test_url_with_value():
    assert _parse_args(["--url", "http://127.0.0.1:9000/"]) == ("http://127.0.0.1:9000/", False)


def test_trailing_url_falls_back_to_default():
    assert _parse_args(["--url"]) == (DEFAULT_URL, False)


def test_url_never_swallows_a_flag():
    assert _parse_args(["--url", "--on-top"]) == (DEFAULT_URL, True)


def test_host_port_defaults():
    assert _host_port("http://127.0.0.1:8000/") == ("127.0.0.1", 8000)
    assert _host_port("https://example.test/") == ("example.test", 443)


def test_out_of_range_port_falls_back_to_default():
    assert _parse_args(["--url", "http://127.0.0.1:99999/"]) == (DEFAULT_URL, False)


def test_garbage_url_falls_back_to_default():
    assert _parse_args(["--url", "not a url"]) == (DEFAULT_URL, False)
