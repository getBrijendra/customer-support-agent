"""Policy checks for write capabilities, enforced in code (never in the prompt).

Binding policy keys:
- `max_amount`: upper bound for the `amount` argument
- `max_pct`: upper bound for the `percent` argument
- `allowed_values`: {arg: [allowed values]}
- `blocked`: true disables writes for that capability entirely
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.capabilities.registry import Capability


@dataclass
class PolicyCheck:
    ok: bool
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "violations": self.violations}


def check_policy(cap: Capability, args: dict[str, Any], policy: dict[str, Any]) -> PolicyCheck:
    if cap.kind != "write":
        return PolicyCheck(ok=True)
    violations: list[str] = []
    if policy.get("blocked"):
        violations.append(f"{cap.name} is blocked by policy")
    if "amount" in args and float(args["amount"]) <= 0:
        violations.append("amount must be positive")
    max_amount = policy.get("max_amount")
    if max_amount is not None and "amount" in args and float(args["amount"]) > float(max_amount):
        violations.append(f"amount {args['amount']} exceeds max_amount {max_amount}")
    max_pct = policy.get("max_pct")
    if max_pct is not None and "percent" in args and float(args["percent"]) > float(max_pct):
        violations.append(f"percent {args['percent']} exceeds max_pct {max_pct}")
    for arg, allowed in (policy.get("allowed_values") or {}).items():
        if arg in args and args[arg] not in allowed:
            violations.append(f"{arg}={args[arg]!r} is not an allowed value")
    return PolicyCheck(ok=not violations, violations=violations)
