"""
nanomem.errors
~~~~~~~~~~~~~~
Exception hierarchy for the v3 container.

Every error raised by :mod:`nanomem.container`, :mod:`nanomem.crypto` and
:mod:`nanomem.engine` derives from :class:`NanomemError`, so a caller can guard
the whole library with a single ``except NanomemError``.
"""


class NanomemError(Exception):
    """Base class for every nanomem-specific error."""


class CorruptContainerError(NanomemError):
    """The file header is missing, truncated, or not a nanomem container."""


class IntegrityError(NanomemError):
    """A complete block failed its trailer/MAC check.

    Attributes
    ----------
    block_index : int   ordinal of the offending block within the file
    offset      : int   byte offset of the block header
    reason      : str   short human-readable cause
    """

    def __init__(self, block_index: int, offset: int, reason: str = "trailer mismatch"):
        self.block_index = int(block_index)
        self.offset = int(offset)
        self.reason = str(reason)
        super().__init__(
            f"block {self.block_index} at offset {self.offset}: {self.reason}"
        )


class WrongPasswordError(NanomemError):
    """The supplied password does not authenticate the file header."""


class PasswordRequiredError(NanomemError):
    """The file is encrypted but no password was supplied."""


class ReadOnlyVaultError(NanomemError):
    """A write was attempted on a vault opened read-only (or an un-migrated v2 file)."""


class ContainerReplacedError(NanomemError):
    """The file on disk was replaced (new inode or new vault_uuid) under us."""


class EmbeddingWidthError(NanomemError):
    """A declared ``dim=`` disagrees with the width the endpoint actually returns.

    ``EmbeddingProvider(dim=N)`` is documented as a way to skip the startup
    probe. Through 0.7.10 it was taken purely on trust: ``.dim`` reported N
    while ``embed()`` returned whatever the endpoint sent, so the object
    advertised one width and produced another, and a caller sizing anything
    from ``.dim`` was sizing it wrong.
    """


class NotEncryptedError(NanomemError):
    """A password was supplied for a vault that is stored in plaintext.

    A subclass of :class:`NanomemError` on purpose: 3.0.1 raised a bare
    ``ValueError`` here, which escaped every documented ``except NanomemError``
    guard -- including the CLI's, which printed a raw traceback for
    ``--password`` against a plaintext vault.
    """


class VaultShrankError(NanomemError):
    """The vault file lost bytes an open mapping still addresses.

    Raised by the vault-backed arena layouts (``arena_cache_vectors="offsets"``
    / ``"offsets_ram"``, ``arena_cache_records="vault"``), which read vectors
    and record sections straight out of a read-only ``mmap`` of the vault
    instead of out of a copy in the sidecar. If the file is truncated by
    something outside this process while that mapping is live, touching a page
    that is now past EOF raises SIGBUS -- a signal, not an exception, which
    kills the process with no traceback and no chance to recover. The layouts
    that copy the bytes into the sidecar (``"cache"``) have their own copy and
    survive the same damage, so this failure mode is the price of not copying
    and it is named here rather than left as a crash.

    The check that raises this is a size comparison against the last byte the
    block table can address, so nanomem's OWN torn-tail recovery -- which only
    removes bytes PAST the last valid block -- does not trigger it. It is a
    guard, not a guarantee: a truncation that lands between the check and the
    page touch still faults.
    """


class ClosedVaultError(NanomemError):
    """A write was attempted after ``close()``.

    3.0.1 accepted it. On a password-protected vault the keys are wiped by
    ``close()``, so the record was written in CLEARTEXT into a container whose
    header still claimed ENCRYPTED, and the vault could never be opened again.
    """


__all__ = [
    "NanomemError",
    "CorruptContainerError",
    "IntegrityError",
    "WrongPasswordError",
    "PasswordRequiredError",
    "ReadOnlyVaultError",
    "ContainerReplacedError",
    "NotEncryptedError",
    "ClosedVaultError",
]
