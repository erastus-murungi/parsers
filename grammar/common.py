import re


def group(*choices):
    return "(" + "|".join(choices) + ")"


def maybe(*choices):
    return group(*choices) + "?"


# Regular expressions used to parse numbers
# borrowed from python tokenizer library
Hexnumber = r"0[xX](?:_?[0-9a-fA-F])+"
Binnumber = r"0[bB](?:_?[01])+"
Octnumber = r"0[oO](?:_?[0-7])+"
Decnumber = r"(?:0(?:_?0)*|[1-9](?:_?[0-9])*)"
Intnumber = maybe(r"[+-]") + group(Hexnumber, Binnumber, Octnumber, Decnumber)
Exponent = r"[eE][-+]?[0-9](?:_?[0-9])*"
Pointfloat = group(
    r"[0-9](?:_?[0-9])*\.(?:[0-9](?:_?[0-9])*)?", r"\.[0-9](?:_?[0-9])*"
) + maybe(Exponent)
Expfloat = r"[0-9](?:_?[0-9])*" + Exponent
Floatnumber = maybe(r"[+-]") + group(Pointfloat, Expfloat)
Imagnumber = group(r"[0-9](?:_?[0-9])*[jJ]", Floatnumber + r"[jJ]")


# borrowed from https://github.com/lark-parser/lark/blob/master/lark/grammars/common.lark
_STRING_INNER = r".*?"
_STRING_ESC_INNER = _STRING_INNER + r"(?<!\\)(\\\\)*?"
ESCAPED_STRING = '"' + _STRING_ESC_INNER + '"'


common_patterns = {
    "integer": re.compile(Intnumber),
    "float": re.compile(Floatnumber),
    "whitespace": re.compile(r"\s+"),
    "newline": re.compile(r"\n"),
    "char": re.compile(r"."),
    "word": re.compile(r"\w+"),
    "escaped_string": re.compile(ESCAPED_STRING, re.DOTALL),
    "number": re.compile(Floatnumber + "|" + Intnumber),
}
