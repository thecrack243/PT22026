"""Local launcher: works around a malformed certificate in the Windows store."""
import ssl
import sys

_original = ssl.SSLContext._load_windows_store_certs


def _safe_load(self, storename, purpose):
    try:
        return _original(self, storename, purpose)
    except ssl.SSLError:
        return bytearray()          # ignore the unreadable certificate store


ssl.SSLContext._load_windows_store_certs = _safe_load

from streamlit.web import cli as stcli  # noqa: E402

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app/app.py"]
    sys.exit(stcli.main())