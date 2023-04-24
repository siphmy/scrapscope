
# scrapscope

## Getting started

### Prerequisites

- Docker
- Python 3.11.3

### Installation

#### With venv

- Create environment

  ```shell
  python -m venv .venv
  ```

- Activate environment

  ```shell
  source .venv/bin/activate
  ```

  cf. <https://docs.python.org/3/library/venv.html#how-venvs-work>

- Install packages

  ```shell
  pip install -r requirements.txt
  ```

### Usage

#### Launch database

```shell
docker compose up
```

#### Sync project

```shell
./scrapscope sync <path to project.json>
```

> **Note**
>
> The filename without extension will be adopted as its corresponding index name.
> e.g. the index for `foo/bar.json` will be named `bar`.

#### Search

```shell
./scrapscope search <index>
```

#### List indices

```shell
./scrapscope list
```
