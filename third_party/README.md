# Third-party source and patch policy

This directory never contains an unreviewed `cp -a` of a developer's working
directory. Every entry must have a provenance record with source URL, commit,
license, redistribution status and content hash.

The current release uses the following strategy:

- Public, license-cleared source may be imported as a subtree.
- The custom LeRobot SONIC delta is represented by a patch against a pinned
  upstream base; test-only LFS data is not migrated.
- The custom G1 simulator delta is represented by a patch. Robot mesh assets
  are downloaded by the operator because their redistribution license is not
  established.
- NVIDIA GEAR-SONIC/GR00T weights remain external, revision-pinned artifacts.
