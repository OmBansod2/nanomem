"""
nanomem.users
~~~~~~~~~~~~~
User profile management for NanoMem.
Handles profile creation, listing, switching, and deletion with strict data isolation.
"""

import os
import re
import glob
from typing import List, Dict, Any, Optional
from .vault import Vault


def get_vault_filename(username: str) -> str:
    """Derive standard vault filename from username with strict sanitization."""
    raw = username.strip()
    basename = os.path.basename(raw)
    if basename.endswith(".dat"):
        base_clean = basename[:-4]
    else:
        base_clean = basename
    clean = re.sub(r"[^\w\-_]", "", base_clean)
    if not clean:
        clean = "user"
    if clean.lower() == "default":
        return "memory.dat"
    return f"memory_{clean}.dat"


def list_users(directory: str = ".") -> List[Dict[str, Any]]:
    """Scan and list all created user profiles and their vault stats."""
    users = []
    
    # 1. Default vault (memory.dat)
    default_path = os.path.join(directory, "memory.dat")
    if os.path.exists(default_path):
        users.append(_build_user_info("default", default_path))

    # 2. Named user vaults (memory_*.dat). `.v2.bak` / `.tmp-*` siblings are not
    #    users -- the glob only matches `*.dat`, and a `memory_x.dat.v2.bak` does
    #    not, but be explicit about it so a future glob change cannot regress.
    pattern = os.path.join(directory, "memory_*.dat")
    for fpath in sorted(glob.glob(pattern)):
        fname = os.path.basename(fpath)
        if fname.endswith(".v2.bak") or ".tmp-" in fname:
            continue
        uname = fname[7:-4]  # Strip 'memory_' and '.dat'
        if uname and uname != "default":
            users.append(_build_user_info(uname, fpath))

    return users


def _build_user_info(username: str, filepath: str) -> Dict[str, Any]:
    size_kb = os.path.getsize(filepath) / 1024.0 if os.path.exists(filepath) else 0.0
    doc_count = 0
    try:
        with Vault(filepath) as v:
            doc_count = v.stats().get("total_documents", 0)
    except Exception:
        pass

    return {
        "username": username,
        "filename": os.path.basename(filepath),
        "filepath": os.path.abspath(filepath),
        "size_kb": round(size_kb, 1),
        "documents": doc_count
    }


def user_exists(username: str, directory: str = ".") -> bool:
    """Check if a specific user profile vault exists."""
    target = os.path.join(directory, get_vault_filename(username))
    return os.path.exists(target)


def create_user(username: str, directory: str = ".", password: Optional[str] = None) -> str:
    """Create a new isolated user profile vault (plaintext unless ``password``)."""
    clean = username.strip()
    sanitized = re.sub(r"[^\w\-_]", "", os.path.basename(clean))
    if not clean or not sanitized:
        raise ValueError("Username cannot be empty or contain only invalid path characters.")
    
    target = os.path.join(directory, get_vault_filename(clean))
    if os.path.exists(target):
        raise FileExistsError(f"User '{clean}' already exists at '{target}'.")

    # Write the (PLAINTEXT by default) vault header on disk. The old comment here
    # said "encrypted"; `Vault(path)` with no password creates a plaintext file
    # and `stats()['encrypted_at_rest']` says so. Pass a password through
    # `Vault(..., password=...)` or NANOMEM_PASSWORD for the encrypted mode.
    with Vault(target, password=password) as v:
        pass

    return target


def delete_user(username: str, directory: str = ".") -> bool:
    """Delete a user profile vault AND its siblings (DECISIONS #12).

    ``<path>.v2.bak`` is a byte-identical PLAINTEXT copy of a migrated v2 vault
    and ``<path>.tmp-*`` is a partial rewrite; 3.0.1 removed only ``<path>`` and
    left both behind, so "delete this user" left the user's memories readable on
    disk. Returns True when the vault itself existed.
    """
    clean = username.strip()
    target = os.path.join(directory, get_vault_filename(clean))
    found = os.path.exists(target)
    for p in [target, target + ".v2.bak"] + sorted(glob.glob(target + ".tmp-*")):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
    return found


def user_vault_siblings(username: str, directory: str = ".") -> List[str]:
    """Every file `delete_user` would remove (for tests and for `nanomem users`)."""
    target = os.path.join(directory, get_vault_filename(username.strip()))
    return [p for p in [target, target + ".v2.bak"] + sorted(glob.glob(target + ".tmp-*"))
            if os.path.exists(p)]
