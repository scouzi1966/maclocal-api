from ledger.core import Ledger
from ledger.money import Money

def test_balance():
    lg = Ledger()
    lg.credit('a', Money(10))
    assert lg.balance('a').cents == 10

def test_allocate_preserves_total():
    lg = Ledger()
    parts = lg.allocate(Money(100), [1, 1, 1])
    assert sum(p.cents for p in parts) == 100
