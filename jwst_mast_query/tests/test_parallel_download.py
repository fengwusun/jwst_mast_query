#! /usr/bin/env python

"""Tests for downloading products in parallel. These tests are unit tests, not
regression tests: the products are served by a local http server, so that no
MAST token is needed and they can also run on Github Actions.

Author
------

F. Sun
"""
import concurrent.futures
import functools
import http.server
import os
import threading

import pytest

from jwst_mast_query.jwst_download import download_product_url
from jwst_mast_query.jwst_download import download_mast
from jwst_mast_query.pdastro import makepath


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that does not print every request to stderr."""
    def log_message(self, format, *args):
        pass


@pytest.fixture
def local_server(tmp_path):
    """Serve tmp_path over http on a free port, for the duration of one test."""
    handler = functools.partial(QuietHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield(f'http://127.0.0.1:{server.server_address[1]}', tmp_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_nprocs_option():
    """The number of parallel downloads is a command line option, default 1."""
    parser = download_mast().define_options()
    assert parser.parse_args([]).nprocs == 1
    assert parser.parse_args(['--nprocs', '8']).nprocs == 8


def test_download_product_url(local_server):
    """A single product is downloaded, and no .part file is left behind."""
    url_base, srcdir = local_server
    content = b'0123456789' * 1000
    (srcdir / 'jw00000001001_00001_00001_nrcalong_uncal.fits').write_bytes(content)

    outfile = srcdir / 'out' / 'obsnum1' / 'jw00000001001_00001_00001_nrcalong_uncal.fits'
    nbytes = download_product_url(f'{url_base}/{outfile.name}', str(outfile))

    assert nbytes == len(content)
    assert outfile.read_bytes() == content
    assert not os.path.exists(str(outfile) + '.part')


def test_download_product_url_missing_file(local_server):
    """A product that MAST does not have is not retried and leaves nothing behind."""
    url_base, srcdir = local_server
    outfile = srcdir / 'out' / 'missing_uncal.fits'

    with pytest.raises(Exception):
        download_product_url(f'{url_base}/missing_uncal.fits', str(outfile))

    assert not outfile.exists()
    assert not os.path.exists(str(outfile) + '.part')


def test_parallel_download(local_server):
    """Many products downloaded at the same time all arrive intact.

    The downloads write into the same directory, which does not exist yet: the
    directory is created once by makepath(), by all the threads at the same time.
    """
    url_base, srcdir = local_server
    nfiles = 8
    contents = {}
    for i in range(nfiles):
        name = f'jw0000000100{i}_00001_00001_nrcalong_uncal.fits'
        contents[name] = bytes([i]) * (100 * 1000)
        (srcdir / name).write_bytes(contents[name])

    outdir = srcdir / 'out' / 'obsnum1'
    with concurrent.futures.ThreadPoolExecutor(max_workers=nfiles) as executor:
        futures = {executor.submit(download_product_url, f'{url_base}/{name}', str(outdir / name)): name
                   for name in contents}
        nbytes = sum(future.result() for future in concurrent.futures.as_completed(futures))

    assert nbytes == sum(len(content) for content in contents.values())
    for name, content in contents.items():
        assert (outdir / name).read_bytes() == content
        assert not os.path.exists(str(outdir / name) + '.part')


def test_makepath_is_thread_safe(tmp_path):
    """makepath() does not raise if several threads create the same path at once."""
    path = str(tmp_path / 'obsnum1' / 'a')

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(makepath, path) for _ in range(8)]

    assert all(f.result() == 0 for f in futures)
    assert os.path.isdir(path)
