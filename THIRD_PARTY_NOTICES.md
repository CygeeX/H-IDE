# Third-party notices

This distribution is an integration workspace. It contains components with
different upstream licenses; do not infer a single license for all source
trees.

## Component policy

- Each imported component retains its upstream notices where present.
- `optional/DAAAM/` records the pinned upstream source and its dependency
  restrictions. Its research/non-commercial weights and datasets are not
  redistributed here.
- `third_party/` contains only source or patch material that has an explicit
  provenance entry. Assets with unclear redistribution rights are fetched by
  the operator after accepting the upstream terms.
- Office V2 furniture assets retain the existing CC0/license files from the
  simulator component.
- The four leader-demo PNG files are covered by
  `components/h-ide/apps/control-pwa/public/assets/leader-demo/ASSET_LICENSE.md`
  and the exact SHA-256 values in the adjacent asset manifest. Company names
  on display-only cards do not imply endorsement or affiliation.
- FARM is not a runtime dependency in this release. The proposed Provider is
  pinned only as provenance; upstream FARM is AGPL-3.0-or-later and its model
  and dependency terms require a separate commercial review.
- DAAAM remains an optional post-Mission, observation-only import contract.
  DAAAM core is BSD-3-Clause, but its default dependency and model stack also
  includes AGPL and non-commercial terms; those runtimes and weights are not
  redistributed here.

Repository visibility does not grant redistribution rights. Before each public
release or deployment, run the license/SBOM checks and obtain approval for every
external model, mesh, dataset, and runtime binary. The current release keeps
uncleared meshes, weights, datasets, and runtime caches outside Git.

The current Git history contains the preserved, unsquashed LeRobot subtree
history. Historical upstream paths can include test media or model-like file
names even when those blobs are absent from the release tree. The release
manifest distinguishes current-tree contents from historical provenance; a
history rewrite must never be performed silently on the public repository.
