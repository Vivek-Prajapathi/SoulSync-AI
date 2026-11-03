import { readdirSync, readFileSync, copyFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const frontendDirectory = dirname(fileURLToPath(import.meta.url));
const projectDirectory = resolve(frontendDirectory, "..");

const modelAssets = {
    "models/movenet_thunder.tflite": resolve(projectDirectory, "movenet_thunder.tflite"),
    "models/pose_cls_xgboost.onnx": resolve(frontendDirectory, "models/pose_cls_xgboost.onnx"),
    "models/pose_cls_xgboost.json": resolve(frontendDirectory, "models/pose_cls_xgboost.json")
};

function copyDirectoryContents(sourceDirectory, destinationDirectory) {
    mkdirSync(destinationDirectory, { recursive: true });
    for (const fileName of readdirSync(sourceDirectory)) {
        copyFileSync(
            resolve(sourceDirectory, fileName),
            resolve(destinationDirectory, fileName)
        );
    }
}

function getRuntimeAssetSources() {
    const sources = new Map();
    sources.set(
        "models/movenet_thunder.tflite",
        resolve(projectDirectory, "movenet_thunder.tflite")
    );
    const tfliteWasmDirectory = resolve(
        frontendDirectory,
        "node_modules/@tensorflow/tfjs-tflite/wasm"
    );
    for (const fileName of readdirSync(tfliteWasmDirectory)) {
        sources.set(fileName, resolve(tfliteWasmDirectory, fileName));
    }

    const onnxRuntimeDirectory = resolve(
        frontendDirectory,
        "node_modules/onnxruntime-web/dist"
    );
    for (const extension of [".mjs", ".wasm"]) {
        const fileName = `ort-wasm-simd-threaded${extension}`;
        sources.set(fileName, resolve(onnxRuntimeDirectory, fileName));
    }
    return sources;
}

function browserRuntimeAssets() {
    return {
        name: "soulsync-browser-runtime-assets",
        configureServer(server) {
            const runtimeAssets = getRuntimeAssetSources();
            server.middlewares.use((request, response, next) => {
                if (!request.url) {
                    next();
                    return;
                }
                const pathname = new URL(request.url, "http://localhost").pathname;
                const assetName = pathname.replace(/^\/(?:wasm\/)?/, "");
                const assetPath = runtimeAssets.get(assetName);
                if (!assetPath) {
                    next();
                    return;
                }
                response.setHeader(
                    "Content-Type",
                    assetName.endsWith(".wasm") ? "application/wasm" : "text/javascript"
                );
                response.end(readFileSync(assetPath));
            });
        },
        generateBundle() {
            for (const [fileName, sourcePath] of Object.entries(modelAssets)) {
                this.emitFile({
                    type: "asset",
                    fileName,
                    source: readFileSync(sourcePath)
                });
            }
        },
        closeBundle() {
            const outputDirectory = resolve(frontendDirectory, "dist");
            const wasmDirectory = resolve(outputDirectory, "wasm");
            mkdirSync(wasmDirectory, { recursive: true });

            const tfliteWasmDirectory = resolve(
                frontendDirectory,
                "node_modules/@tensorflow/tfjs-tflite/wasm"
            );
            copyDirectoryContents(tfliteWasmDirectory, outputDirectory);

            const onnxRuntimeDirectory = resolve(
                frontendDirectory,
                "node_modules/onnxruntime-web/dist"
            );
            for (const extension of [".mjs", ".wasm"]) {
                const fileName = `ort-wasm-simd-threaded${extension}`;
                copyFileSync(
                    resolve(onnxRuntimeDirectory, fileName),
                    resolve(wasmDirectory, fileName)
                );
            }
        }
    };
}

export default defineConfig({
    root: frontendDirectory,
    base: "/",
    plugins: [browserRuntimeAssets()],
    resolve: {
        conditions: ["onnxruntime-web-use-extern-wasm"]
    },
    build: {
        outDir: resolve(frontendDirectory, "dist"),
        emptyOutDir: true,
        rollupOptions: {
            input: {
                home: resolve(frontendDirectory, "index.html"),
                session: resolve(frontendDirectory, "session.html")
            }
        }
    }
});
