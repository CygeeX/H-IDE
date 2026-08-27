import { loadEnv, type ProxyOptions } from "vite";
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { SCENE_API_PATH } from "./src/vite-proxy-routing";

function sceneProxy(target: string, token: string): ProxyOptions {
  return {
    target,
    changeOrigin: true,
    configure(proxy) {
      proxy.on("proxyReq", (proxyRequest) => {
        if (token) proxyRequest.setHeader("authorization", `Bearer ${token}`);
      });
    },
  };
}

export default defineConfig(({ mode }) => {
  const environment = loadEnv(mode, ".", "");
  const target = environment.VITE_DEV_PROXY_TARGET ?? "http://127.0.0.1:8080";
  const sceneTarget = environment.HIDE_DEV_SCENE_API_TARGET?.trim();
  const sceneToken = environment.HIDE_DEV_SCENE_API_TOKEN?.trim() ?? "";
  const directSceneProxy = sceneTarget
    ? sceneProxy(sceneTarget, sceneToken)
    : { target, changeOrigin: true };
  return {
    plugins: [react()],
    server: {
      proxy: {
        ...(sceneTarget ? { [SCENE_API_PATH.source]: directSceneProxy } : {}),
        "/scene-assets": directSceneProxy,
        "/api": { target, changeOrigin: true, ws: true },
      },
    },
    test: {
      environment: "jsdom",
      setupFiles: ["./src/test-setup.ts"],
      css: true,
    },
  };
});
