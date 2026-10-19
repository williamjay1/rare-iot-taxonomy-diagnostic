# Porting the archived path contract

The experiment scripts were written against a frozen Windows workspace. Absolute paths
were replaced during release preparation to avoid exposing local directories.

Placeholders:

- `<WORKSPACE>` or `workspace`: the experiment workspace containing `harmonized/`,
  `extensions/`, `temporal/`, `external_grouped/`, `pseudo_null/`, `standards/`, and
  `primary_audit/`.
- `<LEGACY_PROJECT>` or `legacy_project`: the prior project containing `scripts/` and
  the legacy data cache.
- `<RELEASE>` or `release`: this repository root.
- `<HOME>` or `home`: the user home directory.

Some scripts import modules from the legacy project or expect cached prediction arrays.
Those arrays are not included. Adjust `sys.path`, `BASE`, `ROOT`, or `PROJECT` constants
at the top of each script to the local layout before execution.

The scripts are provided for audit and re-execution with the licensed raw data. They are
not a supported software product.
