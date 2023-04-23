import argparse
import sys
from argparse import ArgumentError, ArgumentParser, Namespace

from . import config, database, index, models, scrapbox


def sync(*, args: Namespace, idx: index.Index):
    project = scrapbox.Project(file=args.file)
    idx.index(project, force=args.force)


def search(*, args: Namespace, idx: index.Index):
    idx.model.preload()

    while True:
        try:
            prompt = input("query > ")
            hits = idx.query(prompt)
            for hit in hits:
                doc = hit.document
                print(f"[{hit.score:.04f}] {doc.page_title}: {doc.content}")
        except (EOFError, KeyboardInterrupt):
            break


def arg_parser():
    parser = ArgumentParser(
        "scrapscope",
        description="semantic search for scrapbox",
    )
    subparsers = parser.add_subparsers(required=True)

    parser_sync = subparsers.add_parser(
        "sync",
        description="sync index with scrapbox project",
    )
    parser_sync.add_argument("file")
    parser_sync.add_argument(
        "-f",
        help="forces recalculation of embeddings",
        dest="force",
        action=argparse.BooleanOptionalAction,
    )
    parser_sync.set_defaults(handler=sync)

    parser_search = subparsers.add_parser(
        "search",
        description="launch interactive prompt for searching pages",
    )
    parser_search.set_defaults(handler=search)

    return parser


if __name__ == "__main__":
    parser = arg_parser()
    try:
        args = parser.parse_args()
    except ArgumentError:
        parser.print_help()
        sys.exit(1)

    db = database.Qdrant(
        host=config.qdrant_host,
        port=config.qdrant_port,
        collection=config.qdrant_collection,
    )
    model = models.STParaphraseMultilingualMiniLmL12V2()
    idx = index.Index(db=db, model=model)

    args.handler(args=args, idx=idx)
