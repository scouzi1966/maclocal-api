from pkg15.service import run

def test_run():
    assert run(2) == 4

def test_import_api():
    from pkg15 import api
    assert api
