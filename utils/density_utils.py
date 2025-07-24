def compute_density(count, total=None, weight=1.0):
    if total:
        return round(count / total, 2) if total > 0 else 0.0
    return round(count * weight, 2)
