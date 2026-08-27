import { configDefaults, defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    testTimeout: 5_000,
    // `mastra build` installs a self-contained runtime under this ignored
    // directory.  Its transitive packages may contain their own test files;
    // they are build artifacts, not this repository's test suite.
    exclude: [...configDefaults.exclude, ".mastra/**"],
  },
});
