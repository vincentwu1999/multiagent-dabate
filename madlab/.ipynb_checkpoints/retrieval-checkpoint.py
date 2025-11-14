import wikipedia
from .utils import truncate

def wiki_search_and_snippets(query: str, k: int = 2, char_limit: int = 1500) -> str:
    try:
        wikipedia.set_lang("en")
        hits = wikipedia.search(query, results=k)
        snippets = []
        for h in hits:
            try:
                pg = wikipedia.page(h, auto_suggest=False)
                if pg.summary:
                    snippets.append(f"Title: {pg.title}\nSummary:\n{pg.summary}")
            except Exception:
                continue
        joined = "\n\n".join(snippets)
        return truncate(joined, max_chars=char_limit) if joined else ""
    except Exception:
        return ""
