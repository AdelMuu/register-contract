from dataclasses import dataclass

@dataclass(frozen=True)
class CheckResult:
    ok: bool
    reason: str
    
    @staticmethod
    def pass_(reason: str = "contract satisfied"):
        return CheckResult(True, reason)

    @staticmethod
    def fail(reason: str):
        return CheckResult(False, reason)
