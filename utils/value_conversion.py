import re

def value_to_float(x):
    if isinstance(x, (int, float)):
        return x

    x = x.replace('£', '').strip()

    if '-' in x:
        parts = x.split('-')
        try:
            low = convert_single_value(parts[0].strip())
            high = convert_single_value(parts[1].strip())
            return (low + high) / 2
        except:
            return 0.0
    else:
        return convert_single_value(x)

def convert_single_value(val):
    if 'M' in val:
        return float(val.replace('M', '')) * 1_000_000
    elif 'K' in val:
        return float(val.replace('K', '')) * 1_000
    else:
        return float(val)
