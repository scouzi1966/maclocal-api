from ledger.money import Money

class Ledger:
    def __init__(self):
        self._entries = []
    def credit(self, account: str, amount: Money):
        self._entries.append((account, amount.cents))
    def balance(self, account: str) -> Money:
        total = sum(c for a, c in self._entries if a == account)
        return Money(total)
    def allocate(self, total: Money, ratios):
        total_ratio = sum(ratios)
        parts = []
        allocated = 0
        for i, r in enumerate(ratios):
            if i == len(ratios) - 1:
                parts.append(Money(total.cents - allocated))
            else:
                part = int(total.cents * r / total_ratio)
                parts.append(Money(part))
                allocated += part
        return parts
