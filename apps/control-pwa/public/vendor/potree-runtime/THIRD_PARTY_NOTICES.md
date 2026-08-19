# Potree runtime notices

This directory contains the minimal browser runtime built from Potree 1.8.2
(source revision `5636cd47`) and the JavaScript libraries bundled by that build.

Potree is licensed under the BSD 2-Clause license. See [LICENSE](./LICENSE).
The runtime bundle also contains third-party components retained from the
upstream Potree distribution. Source and license details are available in the
upstream Potree repository and its `libs` directory.

Only browser runtime files needed by H-IDE's point-cloud viewer are vendored;
the upstream build system and example assets are not redistributed here. The
included `resources/textures/matcap/matcap.jpg` is Potree's default point-cloud
material texture and is required when `PointCloudMaterial` is constructed.
H-IDE's production-only bundle omits source maps and disables the unused
device-orientation controller and upstream timing/debug messages. Its orbit
controller also has a small H-IDE navigation-mode patch: primary-drag panning
for the locked 2.5D map and simultaneous two-finger pan/zoom on touch screens.
Script-root detection is query-safe so worker and resource URLs remain valid.
