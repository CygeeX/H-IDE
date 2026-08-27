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

Repository visibility does not grant redistribution rights. Before each public
release or deployment, run the license/SBOM checks and obtain approval for every
external model, mesh, dataset, and runtime binary. The current release keeps
uncleared meshes, weights, datasets, and runtime caches outside Git.
