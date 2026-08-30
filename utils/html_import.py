"""Convert Football Manager HTML exports into the CSVs the model reads.

FM writes its squad views as an HTML table. Two things make a naive
`pd.read_html(path)` lossy:

* **No charset declaration.** The file is UTF-8, but with no `<meta charset>`
  the parser guesses, and on a Windows default it decodes as cp1252. On a real
  9,260-row export that silently mangled 1,853 player names
  ("Patryk Stępień" -> "Patryk StÄ\\x99pieÅ\\x84") and broke five whole leagues
  out of the coefficient table -- 298 players dropped from the model for no
  reason anyone would notice. So the encoding is always passed explicitly.

* **Trailing junk rows.** Some exports carry a sigames.com credit inside the
  table. Rather than slicing off a fixed number of rows -- which deletes real
  players when the credit is *outside* the table, as it is in current exports --
  junk rows are detected by what they contain.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def is_mojibake(text) -> bool:
    """True if `text` looks like UTF-8 that was decoded as latin-1/cp1252.

    Tested by round trip rather than by spotting suspicious characters: if the
    string can be re-encoded to latin-1 and then decoded as UTF-8 to give
    something *different*, the original decode was wrong. Character spotting
    gets this wrong on real names -- "Ângelo Sá" and "Åge" are perfectly good
    text that a naive Ã/Å/Â check flags, whereas the round trip rejects them
    because 'Â' + 'n' is not a valid UTF-8 sequence.

    Non-strings are never mojibake. The guard matters: pandas 3.0 changed
    `.astype(str)` to preserve missing values rather than rendering them as the
    string 'nan', so a blank cell arrives here as a float.
    """
    if not isinstance(text, str):
        return False
    try:
        return text.encode('latin-1').decode('utf-8') != text
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


# Columns a squad export must have for the rest of the pipeline to work.
REQUIRED_COLUMNS = ('Name', 'Division', 'Mins')

# Cell contents that mark a row as a footer credit rather than a player.
JUNK_MARKERS = ('sigames', 'http://', 'https://')


class ConversionError(Exception):
    """Raised when an export cannot be turned into a usable table."""


class NotASquadExportError(ConversionError):
    """The file parsed, but it is not a player list."""


class AlreadyConvertedError(ConversionError):
    """The destination CSV exists and overwrite was not requested.

    Separate from other failures so the caller can tell 'you already have
    this' apart from 'this file is broken'.
    """


def read_export(path, encoding: str = 'utf-8', table: int = 0) -> pd.DataFrame:
    """Parse one FM HTML export into a DataFrame.

    `path` may be a filename or an open file-like object, so an upload that
    never touches disk goes through the same parser as a file on disk.

    `encoding` is passed explicitly because the export declares none; see the
    module docstring for what happens when the parser is left to guess.
    """
    name = getattr(path, 'name', None) or str(path)
    if not hasattr(path, 'read'):
        path = Path(path)
        if not path.is_file():
            raise ConversionError(f"No such file: {path}")

    try:
        # flavor is pinned so a parse failure reports itself, rather than
        # falling back to html5lib and complaining about the wrong dependency.
        tables = pd.read_html(path, encoding=encoding, flavor='lxml')
    except ImportError as exc:
        raise ConversionError(
            f"{exc}. Install it with: pip install lxml") from exc
    except LookupError as exc:
        # lxml is fussy about encoding names: 'latin-1' fails, 'latin1' works.
        raise ConversionError(
            f"{exc}. Try 'latin1', 'iso-8859-1' or 'cp1252' "
            "(lxml rejects the hyphenated 'latin-1').") from exc
    except ValueError as exc:
        raise ConversionError(f"No HTML table found in {name}: {exc}") from exc

    if table >= len(tables):
        raise ConversionError(
            f"{name} has {len(tables)} table(s); asked for index {table}")
    return tables[table], len(tables)


def classify_junk_rows(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Split non-player rows into (blank, credit) masks.

    They are reported separately because they mean different things. FM pads
    large exports with rows of empty <td> cells -- a 14MB defender export
    carried 1,928 of them -- whereas a credit row is the sigames.com footer,
    which current exports keep outside the table entirely.
    """
    blank = df.isna().all(axis=1)
    text = df.astype(str)
    credits = pd.Series(False, index=df.index)
    for marker in JUNK_MARKERS:
        credits |= text.apply(
            lambda col: col.str.contains(marker, case=False, na=False)).any(axis=1)
    return blank, credits & ~blank


def find_junk_rows(df: pd.DataFrame) -> pd.Series:
    """Mask selecting every row that is not a player."""
    blank, credits = classify_junk_rows(df)
    return blank | credits


def detect_mojibake(df: pd.DataFrame, columns=('Name', 'Division')) -> int:
    """Count rows whose text shows signs of a wrong-encoding decode."""
    present = [c for c in columns if c in df.columns]
    if not present:
        return 0
    hit = pd.Series(False, index=df.index)
    for column in present:
        hit |= df[column].astype(str).map(is_mojibake)
    return int(hit.sum())


def convert(path, out_dir, encoding: str = 'utf-8', table: int = 0,
            drop_last: int = 0, overwrite: bool = False,
            verbose: bool = True) -> Path:
    """Convert one HTML export to `out_dir/<name>.csv` and return the path.

    drop_last: additionally remove this many trailing rows. Left at 0 by
    default -- current FM exports keep the sigames credit outside the table, so
    a blind slice would delete real players.
    """
    path = Path(path)
    out_dir = Path(out_dir)
    destination = out_dir / f'{path.stem}.csv'
    if destination.exists() and not overwrite:
        raise AlreadyConvertedError(
            f"{destination} already exists; pass --overwrite to replace it")

    df, table_count = read_export(path, encoding=encoding, table=table)
    if table_count > 1 and verbose:
        print(f"  note: {table_count} tables found, using index {table}")
    rows_in = len(df)

    df = clean_table(df, path.name, drop_last=drop_last, encoding=encoding,
                     verbose=verbose)
    rows_out = len(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False, encoding='utf-8')
    if verbose:
        print(f"  {rows_in} rows -> {rows_out} rows, {len(df.columns)} columns")
        print(f"  wrote {destination}")
    return destination


def clean_table(df: pd.DataFrame, name: str = 'export', drop_last: int = 0,
                encoding: str = 'utf-8', verbose: bool = True) -> pd.DataFrame:
    """Validate and strip a parsed export. Shared by the CLI and uploads."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        # Not every HTML file in a Documents folder is a squad export -- a
        # coefficient table or a league summary would otherwise be written to
        # data/ and show up as a bogus position.
        raise NotASquadExportError(
            f"{name} is missing {', '.join(missing)}, so it does not look "
            "like a squad export")

    mangled = detect_mojibake(df)
    if mangled and verbose:
        print(f"  WARNING: {mangled} row(s) look mis-decoded under "
              f"{encoding!r}. Try a different --encoding.")

    blank, credits = classify_junk_rows(df)
    junk = blank | credits
    if junk.any():
        df = df[~junk]
        if verbose:
            if blank.any():
                # FM pads large exports with rows of empty <td> cells.
                print(f"  dropped {int(blank.sum())} blank spacer row(s)")
            if credits.any():
                print(f"  dropped {int(credits.sum())} footer/credit row(s)")
    elif verbose:
        print('  no junk rows inside the table (the sigames credit sits '
              'outside it, so nothing to strip)')

    if drop_last:
        if drop_last >= len(df):
            raise ConversionError(
                f"drop_last={drop_last} would remove all {len(df)} rows")
        df = df.iloc[:-drop_last]
        if verbose:
            print(f"  dropped {drop_last} trailing row(s) as requested")

    if df.empty:
        raise ConversionError(f"{name}: no rows left after cleaning")
    return df


def convert_all(paths, out_dir, **kwargs) -> tuple[list[Path], int]:
    """Convert several exports, reporting each.

    Returns (paths written, number skipped because they already existed), so
    the caller can distinguish 'nothing to do' from 'everything failed'.
    """
    written, existing = [], 0
    for path in paths:
        print(f"=== {Path(path).name} ===")
        try:
            written.append(convert(path, out_dir, **kwargs))
        except AlreadyConvertedError as exc:
            existing += 1
            print(f"  skipped: {exc}")
        except ConversionError as exc:
            print(f"  SKIPPED: {exc}")
    return written, existing
