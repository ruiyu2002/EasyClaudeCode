#!/usr/bin/env python3
"""Entry point: REPL that feeds user input into the LangGraph agent."""
try:
    import readline
    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agent import graph, extract_text


def main():
    history = []
    while True:
        try:
            query = input("\033[36magent >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        result = graph.invoke({"messages": history, "turn_count": 1})

        # result["messages"] contains the full accumulated history
        history = result["messages"]

        final_text = extract_text(history)
        if final_text:
            print(final_text)
        print()


if __name__ == "__main__":
    main()
