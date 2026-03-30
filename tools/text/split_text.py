def split_text(text: str, max_chunk_size: int = 2000):
    chunks = []
    current = []

    for line in text.split("\n"):
        if sum(len(x) for x in current) + len(line) > max_chunk_size:
            chunks.append("\n".join(current))
            current = []
        current.append(line)

    if current:
        chunks.append("\n".join(current))

    return {"chunks": chunks}
