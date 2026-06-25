from pkg3.service import run

def test_run():
    assert run(2) == 4

def test_import_api():
    from pkg3 import api
    assert api
