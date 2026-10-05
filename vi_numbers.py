"""Vietnamese number normalization for ASR scoring: digits in a hypothesis -> spoken-form words.

Vietnamese has several valid spoken readings of the same number (hai nghìn không trăm hai mươi hai / hai nghìn hai
mươi hai / hai không hai hai; tư vs bốn; lẻ vs linh vs không; nghìn vs ngàn; ...), so one fixed converter cannot match
every reference. Two modes:
  default  : the first (standard cardinal) reading of every number
  best     : per number, pick the reading that gives the fewest word errors against the reference (an optimistic
             "format-forgiving" score: a number only counts as an error when no valid reading matches)
Numbers are only rewritten; nothing else in the hypothesis changes.
"""
import itertools
import re

DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
NUM_RE = re.compile(r"(?<![A-Za-z\d])\d+(?:[.,/\-–]\d+)*(?:%|[A-Za-zđ]{1,3}(?![A-Za-z]))?")
UNITS = {"h": ["giờ", "h"], "kg": ["ki lô gam", "kg"], "km": ["ki lô mét", "km"], "m": ["mét", "m"], "ha": ["héc ta", "ha"],
         "cm": ["xen ti mét", "cm"], "mm": ["mi li mét", "mm"], "g": ["gam", "g"], "tr": ["triệu"], "k": ["nghìn", "k"],
         "đ": ["đồng"], "%": ["phần trăm"]}


def _read3(x, first, full, ngan_unused, v):
    """Read 0 <= x < 1000. `full`: not the leading group, so an empty hundreds place may be spoken ('không trăm')."""
    h, t, u = x // 100, x % 100 // 10, x % 10
    out = []
    if h:
        out += [DIGITS[h], "trăm"]
    elif full and (t or u) and not v["omit"]:
        out += ["không", "trăm"]
    if t == 0:
        if u:
            out += ([v["linh"], DIGITS[u]] if (h or full) else [DIGITS[u]])
    elif t == 1:
        out += ["mười"] + ([v["lam"] if u == 5 else DIGITS[u]] if u else [])
    else:
        out += [DIGITS[t], "mươi"]
        if u:
            out.append({1: v["mot"], 4: v["tu"], 5: v["lam"]}.get(u, DIGITS[u]))
    return out


def cardinal(n, v):
    if n == 0:
        return ["không"]
    names = ["", v["ngan"], "triệu", "tỷ"]
    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    out = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g:
            out += _read3(g, i == len(groups) - 1, i != len(groups) - 1, None, v)
            if names[i]:
                out.append(names[i])
    return out


def _variants():
    for ngan, linh, tu, mot, lam, omit in itertools.product(["nghìn", "ngàn"], ["lẻ", "linh", "không"], ["tư", "bốn"],
                                                            ["mốt", "một"], ["lăm", "năm"], [False, True]):
        yield dict(ngan=ngan, linh=linh, tu=tu, mot=mot, lam=lam, omit=omit)


def int_readings(s):
    """Ordered readings of a digit string; the first one is the 'default'."""
    digit_by_digit = " ".join(DIGITS[int(c)] for c in s)
    seen, out = set(), []

    def add(r):
        if r not in seen:
            seen.add(r)
            out.append(r)

    n = int(s)
    if len(s) > 1 and s[0] == "0":
        add(digit_by_digit)
        add(" ".join(cardinal(n, next(_variants()))))
        add("không " + " ".join(cardinal(n, next(_variants()))))
        return out
    if n < 10**12:
        for v in _variants():
            add(" ".join(cardinal(n, v)))
    if len(s) >= 2:
        add(digit_by_digit)
    return out


def _number_part(s):
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):                 # 1.288 -> thousands separator
        return int_readings(s.replace(".", ""))
    if re.fullmatch(r"\d+,\d+", s):                          # 606,2 -> decimal comma
        a, b = s.split(",")
        frac = [" ".join(DIGITS[int(c)] for c in b)] + ([x for x in int_readings(b)[:1]] if b[0] != "0" else [])
        return [f"{x} phẩy {y}" for x in int_readings(a) for y in frac]
    if re.fullmatch(r"\d+", s):
        return int_readings(s)
    return [s]


def readings(expr):
    """All accepted spoken readings of one numeric expression, default first."""
    m = re.fullmatch(r"(.*?)(%|[A-Za-zđ]{1,3})?", expr)
    core, suf = m.group(1), m.group(2)
    if re.fullmatch(r"\d+(?:\.\d{3})*(,\d+)?", core):
        parts_opts = [_number_part(core)]
        seps = []
    elif re.search(r"[\-–]", core):
        parts = re.split(r"[\-–]", core)
        parts_opts, seps = [_number_part(p) for p in parts], [["đến", "tới", ""]] * (len(parts) - 1)
    elif "/" in core:
        parts = core.split("/")
        parts_opts, seps = [_number_part(p) for p in parts], [["tháng", ""], ["năm", ""]][: len(parts) - 1]
    else:
        parts_opts, seps = [_number_part(core)], []
    out = []
    for combo in itertools.product(*parts_opts):
        for sep_choice in itertools.product(*seps) if seps else [()]:
            text = combo[0]
            for sep, nxt in zip(sep_choice, combo[1:]):
                text += (" " + sep if sep else "") + " " + nxt
            out.append(text)
    if suf:
        sufs = UNITS.get(suf.lower() if suf != "%" else "%", [suf.lower()])
        out = [f"{o} {sf}" for sf in sufs for o in out]
    seen, res = set(), []
    for o in out:
        if o not in seen:
            seen.add(o)
            res.append(o)
    return res


def normalize_numbers(hyp, ref=None, errors=None):
    """Rewrite numbers in `hyp`. With `ref` and an `errors(ref, hyp)` function, choose per number the reading with the
    fewest errors (greedy, left to right); otherwise use each number's default reading."""
    spans = [(m.start(), m.end(), readings(m.group(0))) for m in NUM_RE.finditer(hyp)]
    if not spans:
        return hyp
    choice = [0] * len(spans)

    def build():
        out, pos = [], 0
        for (a, b, rd), c in zip(spans, choice):
            out += [hyp[pos:a], rd[c]]
            pos = b
        out.append(hyp[pos:])
        return "".join(out)

    if ref is not None and errors is not None:
        for i, (_, _, rd) in enumerate(spans):
            best, best_e = 0, None
            for c in range(min(len(rd), 400)):
                choice[i] = c
                e = errors(ref, build())
                if best_e is None or e < best_e:
                    best, best_e = c, e
            choice[i] = best
    return build()


if __name__ == "__main__":
    for t in ["2022", "2005", "06", "95%", "20-30%", "1.288", "606,2", "10h", "24", "20/11", "CO2", "1.600"]:
        r = readings(t)
        print(t, "->", r[0], "|", len(r), "readings")
