import argparse
import math
import sys
import urllib.parse
import webbrowser
from argparse import ArgumentError, ArgumentParser, Namespace
from dataclasses import dataclass
from functools import cached_property
from os import path
from typing import List, Optional

from . import config, database, embedding, index, models, scrapbox


def run_import(*, args: Namespace, db: index.Database, model: embedding.Model):
    (project_name, _) = path.splitext(path.basename(args.file))
    idx = index.Index(project=project_name, db=db, model=model)
    project = scrapbox.Project(file=args.file)

    idx.index(project, force=args.force)


def run_search(*, args: Namespace, db: index.Database, model: embedding.Model):
    idx = index.Index(project=args.project, db=db, model=model)
    idx.model.preload()

    CliSearcher(idx=idx, project=args.project).run()


def run_list(*, args: Namespace, db: index.Database, model: embedding.Model):
    for idx in db.indices():
        print(idx)


def run_remote_set(*, args: Namespace, db: index.Database, model: embedding.Model):
    config.set_remote(args.project, args.url)


@dataclass
class CliSearcher:
    idx: index.Index
    project: str

    limit: int = config.default_limit
    last_hits: Optional[List[index.Hit]] = None

    def run(self):
        while True:
            try:
                prompt = input("query > ")
                if prompt.startswith("/"):
                    self.execute(prompt[1:].split())
                else:
                    self.query(prompt)
            except (EOFError, KeyboardInterrupt):
                break

    def query(self, prompt: str):
        hits = list(self.idx.query(prompt, limit=self.limit))
        digits = math.floor(math.log10(len(hits))) + 1
        for i, hit in enumerate(hits):
            print(f"{i+1:0{digits}d}: [{hit.score:.04f}]", end=" ")

            doc = hit.document
            if doc.kind == "line":
                print(f"{doc.page_title}: {doc.content}")
            else:
                print(f"{doc.page_title}")

        self.last_hits = hits

    def execute(self, command: List[str]):
        try:
            args = self.command_parser.parse_args(command)
            if "handler" in args:
                args.handler(args)
            else:
                print("no command specified", file=sys.stderr)
        except ArgumentError:
            self.command_parser.print_help()
        except SystemExit:
            pass

    @cached_property
    def command_parser(self) -> ArgumentParser:
        parser = ArgumentParser("(repl)", add_help=False)
        subparsers = parser.add_subparsers(required=True)

        parser_open = subparsers.add_parser("open", add_help=False)
        parser_open.add_argument("index", type=int, nargs="?", default=1)
        parser_open.set_defaults(handler=self.exec_open)

        parser_set = subparsers.add_parser("set", add_help=False)
        parser_set.add_argument("key")
        parser_set.add_argument("value")
        parser_set.set_defaults(handler=self.exec_set)

        return parser

    def exec_open(self, args: Namespace):
        if self.last_hits is None:
            print("nothing to open; search something first")
            return

        i: int = args.index - 1
        try:
            hit = self.last_hits[i]
        except IndexError:
            print("index out of bounds")
            return

        title = urllib.parse.quote(hit.document.page_title, safe="")
        remote = config.get_remote(self.project)
        webbrowser.open(f"{remote}/{title}")

    def exec_set(self, args: Namespace):
        if args.key == "limit":
            try:
                limit = int(args.value)
            except ValueError:
                print("limit must be a positive integer")
                return

            if limit < 1:
                print("limit must be a positive integer")

            self.limit = int(args.value)

        else:
            print(f"unrecognized key: {args.key}")


def arg_parser():
    parser = ArgumentParser(
        "scrapscope",
        description="semantic search for scrapbox",
    )
    subparsers = parser.add_subparsers(required=True)

    parser_import = subparsers.add_parser(
        "import",
        description="import scrapbox project",
    )
    parser_import.add_argument("file")
    parser_import.add_argument(
        "-f",
        help="forces recalculation of embeddings",
        dest="force",
        action=argparse.BooleanOptionalAction,
    )
    parser_import.set_defaults(handler=run_import)

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

    parser_remote = subparsers.add_parser("remote")
    subparsers_remote = parser_remote.add_subparsers(required=True)
    parser_remote_set = subparsers_remote.add_parser(
        "set",
        description="configure remote",
    )
    parser_remote_set.add_argument("project")
    parser_remote_set.add_argument("url")
    parser_remote_set.set_defaults(handler=run_remote_set)

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
