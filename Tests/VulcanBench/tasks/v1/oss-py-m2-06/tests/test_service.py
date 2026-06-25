from m2pkg6.service import run

def test_run():
    assert run(2) == 4

def test_import_api():
    from m2pkg6 import api
    assert api
