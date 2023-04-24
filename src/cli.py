import argparse
import sys
from argparse import ArgumentError, ArgumentParser, Namespace
from os import path

from . import config, database, embedding, index, models, scrapbox


def run_sync(*, args: Namespace, db: index.Database, model: embedding.Model):
    (project_name, _) = path.splitext(path.basename(args.file))
    idx = index.Index(project=project_name, db=db, model=model)
    project = scrapbox.Project(file=args.file)

    idx.index(project, force=args.force)


def run_search(*, args: Namespace, db: index.Database, model: embedding.Model):
    idx = index.Index(project=args.project, db=db, model=model)
    idx.model.preload()

    while True:
        try:
            prompt = input("query > ")
            hits = idx.query(prompt)
            for hit in hits:
                doc = hit.document
                if doc.kind == "line":
                    print(f"[{hit.score:.04f}] {doc.page_title}: {doc.content}")
                else:
                    print(f"[{hit.score:.04f}] {doc.page_title}")
        except (EOFError, KeyboardInterrupt):
            break


def run_list(*, args: Namespace, db: index.Database, model: embedding.Model):
    for idx in db.indices():
        print(idx)


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
    parser_sync.set_defaults(handler=run_sync)

    parser_search = subparsers.add_parser(
        "search",
        description="launch interactive prompt for searching pages",
    )
    parser_search.add_argument("project")
    parser_search.set_defaults(handler=run_search)

    parser_list = subparsers.add_parser(
        "list",
        description="list projects",
    )
    parser_list.set_defaults(handler=run_list)

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
    )
    model = models.STMultiQaMiniLmL6CosV1()

    args.handler(args=args, db=db, model=model)
