AVAILABLE_ARCHS = ("sm75", "sm120")


def arch_for_capability(major, minor):
    cap = major * 10 + minor
    avail = sorted(int(a[2:]) for a in AVAILABLE_ARCHS)
    pick = max([a for a in avail if a <= cap], default=min(avail))
    return f"sm{pick}"


__all__ = ["AVAILABLE_ARCHS", "arch_for_capability"]
