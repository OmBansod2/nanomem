"""
nanomem - Quick Interactive Demo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run this script to see nanomem in action:
    python demo.py
"""

import os
from nanomem import Vault

vault_file = "my_demo_vault.dat"
if os.path.exists(vault_file):
    os.remove(vault_file)

print("=" * 60)
print("  nanomem: single-file vector + record store, v3 engine")
print("=" * 60)

# 1. Initialize Vault (creates a single local binary file; plaintext by
#    default -- pass password=... for the optional encrypted-at-rest mode)
with Vault(vault_file) as vault:
    print(f"\n[1] Storing knowledge into '{vault_file}'...")
    
    # Store general facts
    vault.add("The staging database server IP is 192.168.1.50", metadata={"env": "staging"})
    vault.add("Standard refund policy is 30 days with receipt", metadata={"department": "billing"})
    
    # Store user personal facts with temporal revisions
    print("[2] Storing user personal fact (Revision 1)...")
    vault.add("My mobile number is 9811111111", metadata={"entity": "phone"}, timestamp=1000.0, revision=1)
    
    print("[3] User changed phone number (Revision 2)...")
    vault.add("My mobile number is 9899999999", metadata={"entity": "phone"}, timestamp=2000.0, revision=2)

    print("\n[4] Querying for current phone number:")
    results = vault.search("What is my mobile number?")
    print("    Result:", results[0]["text"])
    print("    Revision:", results[0]["revision"])
    print("    Source:", results[0]["source"])
    # "score" is cosine plus documented temporal/entity boosts, so it can
    # exceed 1.0; "cosine" is the true stored-vector cosine.
    print("    Score (cosine + boosts):", round(results[0]["score"], 3))
    print("    Cosine:", round(results[0]["cosine"], 3))

    print("\n[5] Querying for historical phone number:")
    results_hist = vault.search("What was my original mobile number?", temporal_direction="historical")
    print("    Result:", results_hist[0]["text"])
    print("    Revision:", results_hist[0]["revision"])

    print("\n[6] Memory Statistics:")
    stats = vault.stats()
    print(f"    • Total documents: {stats['total_documents']}")
    print(f"    • Active Heap RAM: {stats['active_heap_ram_kb']} KB")
    print(f"    • Encrypted at rest: {stats['encrypted_at_rest']}")
    print(f"    • Encryption Cipher: {stats['cipher']}")

if os.path.exists(vault_file):
    os.remove(vault_file)

print("\n" + "=" * 60)
print(" Demo completed successfully!")
print("=" * 60)
