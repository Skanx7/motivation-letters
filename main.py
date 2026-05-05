import sys

from dotenv import load_dotenv

from agents import Orchestrator


def main() -> None:
    load_dotenv()
    url = (
        sys.argv[1]
        if len(sys.argv) > 1 and sys.argv[1].startswith(("http://", "https://"))
        else None
    )
    final_letter = Orchestrator().run(url)
    print(final_letter)


if __name__ == "__main__":
    main()
