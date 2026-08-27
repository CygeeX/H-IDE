# DAAAM optional integration

DAAAM is intentionally optional and asynchronous in this release. H-IDE uses
its post-Mission export only as a coordinate-free semantic-memory prior. It is
not started by the normal workstation launcher and it can never dispatch
Nav2/SONIC/Robot Agent motion.

## Pinned upstream

- Repository: <https://github.com/MIT-SPARK/DAAAM>
- Commit: `ebd6e3b89763849eb7bce7f75e8d08550ea7344a`
- License: BSD-3-Clause for the upstream project; see the upstream
  `THIRD_PARTY_NOTICES.md` for AGPL, non-commercial, proprietary and dataset
  restrictions.

The complete upstream research environment, checkpoints, TensorRT binaries and
datasets are not copied into this repository. Obtain them independently only
after reviewing their terms.

See [`docs/integration/daaam-semantic-memory.md`](../../components/h-ide/docs/integration/daaam-semantic-memory.md)
for the H-IDE importer contract and verification procedure.
