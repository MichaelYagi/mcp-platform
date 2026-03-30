def load_text(text: str | None, file_path: str | None) -> str:
    if text:
        return text
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError("Either 'text' or 'file_path' must be provided.")
