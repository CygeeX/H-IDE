export const SCENE_API_PATH = /^\/api\/v1\/(?:scene(?:-|s(?:\/|$))|robots\/[^/]+\/scene-binding(?:\/|$))/;

export function isSceneApiPath(path: string): boolean {
  return SCENE_API_PATH.test(path.split("?", 1)[0] ?? path);
}
