__version__ = "0.3"

# hashlib.file_digest is 3.11+; polyfill so rebuild works on 3.10.
import hashlib as _hashlib

if not hasattr(_hashlib, "file_digest"):
    class _FileDigest:
        def __init__(self, name):
            self._h = _hashlib.new(name)

        def update(self, data):
            self._h.update(data)

        def hexdigest(self):
            return self._h.hexdigest()

        def digest(self):
            return self._h.digest()

    def _file_digest(fileobj, digest):
        h = _FileDigest(digest)
        while True:
            chunk = fileobj.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
        return h

    _hashlib.file_digest = _file_digest
