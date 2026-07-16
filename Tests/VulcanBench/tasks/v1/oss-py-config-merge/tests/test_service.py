from configlib.service import run

def test_run():
    assert run(2) == 4

def test_import_api():
    from configlib import api
    assert api
