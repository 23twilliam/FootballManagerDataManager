import pandas as pd
import pytest

from utils import html_import
from utils.html_import import ConversionError

HEADERS = ['Inf', 'Rec', 'Name', 'Transfer Value', 'Division', 'Mins', 'CA']


def _html(rows, footer_inside_table=False, encoding='utf-8'):
    """Build a miniature FM-style export. FM declares no charset, so neither do we."""
    body = ''.join(
        '<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>' for row in rows)
    if footer_inside_table:
        body += ('<tr><td colspan="7">'
                 '<a HREF="https://www.sigames.com/">https://www.sigames.com/</a>'
                 '</td></tr>')
    head = '<tr>' + ''.join(f'<th>{h}</th>' for h in HEADERS) + '</tr>'
    return (f'<html><body><table>{head}{body}</table>'
            '<p align="center"><a HREF="https://www.sigames.com/">'
            'https://www.sigames.com/</a></p></body></html>')


def _write(tmp_path, name, rows, **kwargs):
    path = tmp_path / f'{name}.html'
    encoding = kwargs.pop('encoding', 'utf-8')
    path.write_text(_html(rows, **kwargs), encoding=encoding)
    return path


ROWS = [
    ['', '- - -', 'Patryk Stępień', '£1.5M', 'Premier League', '2,298', '150'],
    ['', '- - -', 'Kerim Tüfekçi', '£0', 'Trendyol Süper Lig', '1,900', '132'],
    ['', '- - -', 'Ângelo Sá', '£750K', 'Liga Portugal Betclic', '900', '128'],
]


# --- encoding --------------------------------------------------------------

def test_utf8_names_survive_the_round_trip(tmp_path):
    source = _write(tmp_path, 'Keepers', ROWS)
    out = html_import.convert(source, tmp_path / 'data', verbose=False)
    df = pd.read_csv(out)
    assert list(df['Name']) == ['Patryk Stępień', 'Kerim Tüfekçi', 'Ângelo Sá']
    assert 'Trendyol Süper Lig' in set(df['Division'])


def test_wrong_encoding_is_detected_and_reported(tmp_path, capsys):
    """The real export has no charset declaration, so a bad guess is silent."""
    source = _write(tmp_path, 'Keepers', ROWS)
    html_import.convert(source, tmp_path / 'data', encoding='cp1252',
                        verbose=True)
    assert 'look mis-decoded' in capsys.readouterr().out


def test_correct_encoding_produces_no_warning(tmp_path, capsys):
    source = _write(tmp_path, 'Keepers', ROWS)
    html_import.convert(source, tmp_path / 'data', verbose=True)
    assert 'mis-decoded' not in capsys.readouterr().out


@pytest.mark.parametrize('text', [
    'Ângelo Sá', 'Åge Hareide', 'Patryk Stępień', 'Vyšejšaja Liha',
    'Premier League', '', '£1.5M',
])
def test_legitimate_text_is_not_flagged_as_mojibake(text):
    """A naive Ã/Å/Â character check flags real names; the round trip does not."""
    assert not html_import.is_mojibake(text)


@pytest.mark.parametrize('text', [
    'Patryk StÄ\x99pieÅ\x84', 'Kerim TÃ¼fekÃ§i', 'VyÅ¡ejÅ¡aja Liha',
    'Ligue 1 McDonaldâ\x80\x99s',
])
def test_mis_decoded_text_is_flagged(text):
    assert html_import.is_mojibake(text)


# --- junk rows -------------------------------------------------------------

def test_footer_outside_the_table_is_not_a_row(tmp_path):
    """read_html already excludes it -- blindly slicing rows would lose players."""
    source = _write(tmp_path, 'Keepers', ROWS)
    df = pd.read_csv(html_import.convert(source, tmp_path / 'data', verbose=False))
    assert len(df) == len(ROWS)
    assert 'Ângelo Sá' in set(df['Name'])


def test_footer_inside_the_table_is_dropped(tmp_path):
    source = _write(tmp_path, 'Keepers', ROWS, footer_inside_table=True)
    df = pd.read_csv(html_import.convert(source, tmp_path / 'data', verbose=False))
    assert len(df) == len(ROWS)
    assert not df.astype(str).apply(
        lambda c: c.str.contains('sigames', case=False)).any().any()


def test_find_junk_rows_catches_empty_and_credit_rows():
    df = pd.DataFrame({
        'Name': ['Real Player', None, 'https://www.sigames.com/'],
        'CA': [150, None, None],
    })
    junk = html_import.find_junk_rows(df)
    assert list(junk) == [False, True, True]


def test_drop_last_is_opt_in(tmp_path):
    source = _write(tmp_path, 'Keepers', ROWS)
    kept = pd.read_csv(html_import.convert(source, tmp_path / 'data', verbose=False))
    assert len(kept) == 3, 'nothing should be sliced off by default'

    trimmed = pd.read_csv(html_import.convert(
        source, tmp_path / 'data2', drop_last=1, verbose=False))
    assert len(trimmed) == 2


def test_drop_last_cannot_empty_the_table(tmp_path):
    source = _write(tmp_path, 'Keepers', ROWS)
    with pytest.raises(ConversionError, match='would remove all'):
        html_import.convert(source, tmp_path / 'data', drop_last=99, verbose=False)


# --- file handling ---------------------------------------------------------

def test_output_is_named_after_the_source_file(tmp_path):
    source = _write(tmp_path, 'AttackingMidfielders', ROWS)
    out = html_import.convert(source, tmp_path / 'data', verbose=False)
    assert out.name == 'AttackingMidfielders.csv'


def test_existing_csv_is_not_clobbered_by_default(tmp_path):
    source = _write(tmp_path, 'Keepers', ROWS)
    html_import.convert(source, tmp_path / 'data', verbose=False)
    with pytest.raises(ConversionError, match='already exists'):
        html_import.convert(source, tmp_path / 'data', verbose=False)
    html_import.convert(source, tmp_path / 'data', overwrite=True, verbose=False)


def test_missing_file_raises_clearly(tmp_path):
    with pytest.raises(ConversionError, match='No such file'):
        html_import.convert(tmp_path / 'nope.html', tmp_path / 'data', verbose=False)


def test_file_without_a_table_raises_clearly(tmp_path):
    source = tmp_path / 'empty.html'
    source.write_text('<html><body><p>no table here</p></body></html>',
                      encoding='utf-8')
    with pytest.raises(ConversionError, match='No HTML table'):
        html_import.convert(source, tmp_path / 'data', verbose=False)


def test_convert_all_skips_bad_files_and_continues(tmp_path, capsys):
    good = _write(tmp_path, 'Keepers', ROWS)
    bad = tmp_path / 'broken.html'
    bad.write_text('<html><body>nothing</body></html>', encoding='utf-8')
    written, existing = html_import.convert_all([bad, good], tmp_path / 'data')
    assert [p.name for p in written] == ['Keepers.csv']
    assert existing == 0
    assert 'SKIPPED' in capsys.readouterr().out


def test_encoding_name_lxml_rejects_gives_a_useful_hint(tmp_path):
    """lxml accepts 'latin1' but not 'latin-1'; the raw LookupError is opaque."""
    source = _write(tmp_path, 'Keepers', ROWS)
    with pytest.raises(ConversionError, match="latin1"):
        html_import.convert(source, tmp_path / 'data', encoding='latin-1',
                            verbose=False)


@pytest.mark.parametrize('value', [None, float('nan'), 3.14, 42, True, [], {}])
def test_non_strings_are_never_mojibake(value):
    """pandas 3.0's astype(str) preserves NaN, so blanks arrive here as floats."""
    assert not html_import.is_mojibake(value)


def test_convert_survives_blank_name_and_division_cells(tmp_path):
    rows = ROWS + [['', '- - -', '', '£0', '', '1,200', '90']]
    source = _write(tmp_path, 'Keepers', rows)
    out = html_import.convert(source, tmp_path / 'data', verbose=False)
    assert len(pd.read_csv(out)) == len(rows)


def test_blank_spacer_rows_are_reported_separately_from_credits(tmp_path, capsys):
    """FM pads big exports with rows of empty <td>; that is not a footer."""
    blank = [['', '', '', '', '', '', '']]
    source = _write(tmp_path, 'Keepers', ROWS + blank, footer_inside_table=True)
    html_import.convert(source, tmp_path / 'data', verbose=True)
    out = capsys.readouterr().out
    assert 'blank spacer row' in out
    assert 'footer/credit row' in out


def test_classify_separates_the_two_kinds():
    df = pd.DataFrame({'Name': ['Real', None, 'https://www.sigames.com/'],
                       'CA': [150, None, None]})
    blank, credits = html_import.classify_junk_rows(df)
    assert list(blank) == [False, True, False]
    assert list(credits) == [False, False, True]


def test_already_converted_is_reported_apart_from_a_real_failure(tmp_path):
    """'You already have this' is not the same as 'this file is broken'."""
    source = _write(tmp_path, 'Keepers', ROWS)
    html_import.convert(source, tmp_path / 'data', verbose=False)

    with pytest.raises(html_import.AlreadyConvertedError, match='--overwrite'):
        html_import.convert(source, tmp_path / 'data', verbose=False)

    written, existing = html_import.convert_all([source], tmp_path / 'data')
    assert written == [] and existing == 1


def test_convert_all_counts_existing_and_written_separately(tmp_path):
    first = _write(tmp_path, 'Keepers', ROWS)
    second = _write(tmp_path, 'Strikers', ROWS)
    html_import.convert(first, tmp_path / 'data', verbose=False)
    written, existing = html_import.convert_all([first, second], tmp_path / 'data')
    assert [p.name for p in written] == ['Strikers.csv']
    assert existing == 1


def test_a_non_squad_html_file_is_rejected(tmp_path):
    """A coefficient table would otherwise land in data/ as a bogus position."""
    source = tmp_path / 'CoeffY2.html'
    source.write_text(
        '<html><body><table><tr><th>Pos</th><th>Nation</th></tr>'
        '<tr><td>1st</td><td>England</td></tr></table></body></html>',
        encoding='utf-8')
    with pytest.raises(html_import.NotASquadExportError, match='squad export'):
        html_import.convert(source, tmp_path / 'data', verbose=False)
    assert not (tmp_path / 'data' / 'CoeffY2.csv').exists()


def test_folder_conversion_keeps_squad_exports_and_skips_the_rest(tmp_path, capsys):
    _write(tmp_path, 'Keepers', ROWS)
    (tmp_path / 'CoeffY2.html').write_text(
        '<html><body><table><tr><th>Pos</th></tr><tr><td>1st</td></tr>'
        '</table></body></html>', encoding='utf-8')
    written, _ = html_import.convert_all(
        sorted(tmp_path.glob('*.html')), tmp_path / 'data')
    assert [p.name for p in written] == ['Keepers.csv']
    assert 'squad export' in capsys.readouterr().out


def test_read_export_accepts_an_in_memory_upload(tmp_path):
    """An upload never touches disk, so it must go through the same parser."""
    import io
    source = _write(tmp_path, 'Keepers', ROWS)
    buffer = io.BytesIO(source.read_bytes())
    buffer.name = 'Keepers.html'
    table, count = html_import.read_export(buffer)
    assert count == 1
    assert list(table['Name']) == ['Patryk Stępień', 'Kerim Tüfekçi', 'Ângelo Sá']


def test_upload_errors_name_the_file_not_the_buffer(tmp_path):
    import io
    buffer = io.BytesIO(b'<html><body>nothing here</body></html>')
    buffer.name = 'scouting.html'
    with pytest.raises(ConversionError, match='scouting.html'):
        html_import.read_export(buffer)


def test_clean_table_is_reusable_without_writing_anything(tmp_path):
    source = _write(tmp_path, 'Keepers', ROWS, footer_inside_table=True)
    table, _ = html_import.read_export(source)
    cleaned = html_import.clean_table(table, 'Keepers.html', verbose=False)
    assert len(cleaned) == len(ROWS)
    assert not list(tmp_path.glob('*.csv')), 'clean_table must not write'
