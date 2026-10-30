"""Download ollervo's public games. Python 3.10+, no extra packages needed.

Run: python download_chess_games.py
Options: --site chesscom | lichess | both; --refresh; --output PATH
Chess.com resumes by month. Lichess downloads one complete snapshot.
PGN tags and clock comments are preserved as provided by each service.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


def download(url, destination, accept, username):
    """Stream to a temporary file; publish only a complete response.

    Retry temporary server/network failures, respecting Retry-After seconds.
    A failed request never replaces a previously successful download.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    for attempt in range(5):
        try:
            request = Request(url, headers={
                'User-Agent': f'PersonalChessDataset/1.0 (Chess.com/Lichess user: {username})',
                'Accept': accept,
            })
            with urlopen(request, timeout=120) as response:
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' in content_type:
                    raise ValueError('Received an HTML page instead of game data.')
                with temporary.open('wb') as output:
                    shutil.copyfileobj(response, output)
                expected = response.headers.get('Content-Length')
                if expected and temporary.stat().st_size != int(expected):
                    raise OSError('Incomplete download.')
            # Catch error pages masquerading as successful PGN exports.
            if accept == 'application/x-chess-pgn':
                with temporary.open('rb') as source:
                    prefix = source.read(8192).decode('utf-8-sig', errors='replace').lstrip()
                if prefix and not prefix.startswith('['):
                    raise ValueError('Response is not a PGN export.')
            temporary.replace(destination)
            return
        except (HTTPError, URLError, OSError) as error:
            if isinstance(error, HTTPError) and error.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f'HTTP {error.code} for {url}') from error
            if attempt == 4:
                raise RuntimeError(f'Download failed after 5 attempts: {url}') from error
            delay = 2 ** (attempt + 1)
            if isinstance(error, HTTPError):
                retry_after = error.headers.get('Retry-After', '')
                if retry_after.isdigit():
                    delay = max(delay, int(retry_after))
            if isinstance(error, HTTPError) and error.code == 429:
                delay = max(delay, 60)
            print(f'  Temporary failure; retrying in {delay}s.', flush=True)
            time.sleep(delay)
        finally:
            temporary.unlink(missing_ok=True)


def combine(paths, destination):
    """Rebuild rather than append, so reruns do not duplicate months."""
    temporary = destination.with_suffix('.pgn.part')
    try:
        with temporary.open('wb') as output:
            for path in paths:
                with path.open('rb') as source:
                    shutil.copyfileobj(source, output)
                output.write(b'\n\n')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def chesscom(username, root, refresh):
    folder = root / 'chesscom'
    folder.mkdir(parents=True, exist_ok=True)
    base = f'https://api.chess.com/pub/player/{quote(username, safe="")}/games'
    index = folder / 'archives.json'
    download(base + '/archives', index, 'application/json', username)
    archives = json.loads(index.read_text(encoding='utf-8'))['archives']
    # Restrict downloaded URLs to the expected API paths.
    pattern = re.compile(re.escape(base) + r'/\d{4}/\d{2}$', re.IGNORECASE)
    archives = sorted(set(archives))
    if any(not pattern.fullmatch(url) for url in archives):
        raise ValueError('Unexpected archive URL in API response.')
    now = datetime.now(timezone.utc)
    current_month = f'{now.year:04d}/{now.month:02d}'
    paths = []
    for number, url in enumerate(archives, 1):
        month = '/'.join(url.split('/')[-2:])
        path = folder / (month.replace('/', '-') + '.pgn')
        # Refresh the current month and most recent archive on each run.
        if refresh or not path.exists() or month == current_month or url == archives[-1]:
            print(f'Chess.com [{number}/{len(archives)}]: {month}', flush=True)
            download(url + '/pgn', path, 'application/x-chess-pgn', username)
            time.sleep(1)  # Sequential requests are deliberately gentle on the API.
        else:
            print(f'Chess.com [{number}/{len(archives)}]: {month} already saved')
        paths.append(path)
    result = folder / f'{username}_chesscom.pgn'
    combine(paths, result)
    print(f'Chess.com complete: {result}')


def lichess(username, root, refresh):
    folder = root / 'lichess'
    result = folder / f'{username}_lichess.pgn'
    if result.exists() and not refresh:
        print(f'Lichess already saved: {result} (use --refresh to update)')
        return
    parameters = urlencode({'moves': 'true', 'tags': 'true', 'clocks': 'true',
                            'evals': 'false', 'opening': 'true',
                            'ongoing': 'false', 'finished': 'true'})
    url = f'https://lichess.org/api/games/user/{quote(username, safe="")}?{parameters}'
    print('Lichess: downloading complete history; this may take a few minutes.', flush=True)
    download(url, result, 'application/x-chess-pgn', username)
    print(f'Lichess complete: {result}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--username', default='ollervo')
    parser.add_argument('--site', choices=['chesscom', 'lichess', 'both'], default='both')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / 'chess_data')
    parser.add_argument('--refresh', action='store_true',
                        help='Redownload saved archives, including the Lichess snapshot.')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.username):
        parser.error('Username must contain letters, digits, underscores or hyphens.')
    args.output = args.output.expanduser().resolve() / args.username.lower()
    failures = []
    for site, function in [('chesscom', chesscom), ('lichess', lichess)]:
        if args.site in ('both', site):
            try:
                function(args.username, args.output, args.refresh)
            except (RuntimeError, ValueError, OSError, KeyError) as error:
                print(f'{site}: {error}', file=sys.stderr)
                failures.append(site)
    if failures:
        print('Some downloads failed. Rerun to reuse completed downloads.', file=sys.stderr)
        return 1
    print('Finished. Next step: parse the PGNs and extract only your turns.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped. Completed downloads are retained.', file=sys.stderr)
        raise SystemExit(130)
