# CLI Commands

## `clx config`

This allows you to configure whether the package should automatically load the `.env` file with `python-dotenv`.

```bash
clx config --autoload-env on
```

## `clx manage`

This command is a wrapper around Django's `manage.py` command. It ensures that Django is initialized before running the command.

For example, to run the development server:

```bash
clx manage runserver
```

## Docket Data Sample Generation

The docket sample generation script has been moved out of the CLI and into a standalone script at [`projects/docket_data/run.py`](../../projects/docket_data/run.py). Run it directly with:

```bash
python projects/docket_data/run.py [--import] [--skip-cached-steps]
```
