import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base: "./" keeps asset paths relative, so the site works on a GitHub Pages
// project URL (https://<user>.github.io/<repo>/) without hard-coding the repo name.
export default defineConfig({
  plugins: [react()],
  base: "./",
});
