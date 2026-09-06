import json
import os

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for real-time or up-to-date information. "
                "Use this when the user asks about current events, recent news, "
                "live data, or anything that requires fresh information beyond "
                "your training cutoff."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def _get_exa_key() -> str:
    key = os.environ.get("EXA_API_KEY")
    if not key:
        raise RuntimeError(
            "EXA_API_KEY environment variable is not set. "
            "Export it or add it to your shell profile."
        )
    return key


def execute_tool(name: str, arguments: str) -> str:
    try:
        args = json.loads(arguments)
        if name == "web_search":
            return _web_search(args["query"])
    except Exception as exc:  # noqa: BLE001 - feed failures back to the model
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    return json.dumps({"error": f"Unknown tool: {name}"})


def _web_search(query: str) -> str:
    from exa_py import Exa

    exa = Exa(api_key=_get_exa_key())
    results = exa.search(
        query,
        num_results=5,
        contents={"highlights": {"max_characters": 4000}},
    )

    output = []
    for r in results.results:
        entry = {"title": r.title, "url": r.url}
        if r.highlights:
            entry["highlights"] = r.highlights
        output.append(entry)

    return json.dumps(output, ensure_ascii=False)
