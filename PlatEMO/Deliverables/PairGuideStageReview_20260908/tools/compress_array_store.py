"""Lossless XZ storage inside an ordinary DEFLATE ZIP; verify bytes before removal."""
import lzma
from pathlib import Path
import sys
root=Path(sys.argv[1]);source=root/'arrays.bin';data=source.read_bytes()
encoded=lzma.compress(data,preset=6);assert lzma.decompress(encoded)==data
(root/'arrays.bin.xz').write_bytes(encoded)
source.unlink()  # Only the generated temporary uncompressed store in this package.
print('ARRAY_STORE_XZ_VERIFIED',len(data),'->',len(encoded))
