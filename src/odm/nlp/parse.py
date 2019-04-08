import re

arrow_symbols = ["->", "-->", "..>", "--", "-", ".."]
escaped_arrow_symbols = [re.escape(symbol) for symbol in arrow_symbols]


def lex_veclang(text):

    patterns = "|".join(["{.*?}"] + escaped_arrow_symbols)
    matches = re.findall(patterns, text)
    tokens = [re.sub('[{}]', '', match) for match in matches]

    return tokens


def parse_veclang(text):

    tokens = lex_veclang(text)

    text = [token for token in tokens if token not in arrow_symbols]
    arrows = []

    arrow_instances = [(index, token) for index, token in enumerate(tokens)
                       if token in arrow_symbols]
    for index, token in arrow_instances:
        first = text.index(tokens[index-1])
        second = text.index(tokens[index+1])
        arrows.append((first, second, token))

    return text, arrows
