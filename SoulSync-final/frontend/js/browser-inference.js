import "@tensorflow/tfjs-backend-cpu";
import * as tf from "@tensorflow/tfjs-core";
import * as tflite from "@tensorflow/tfjs-tflite/dist/tf-tflite.fesm.js";
import * as ort from "onnxruntime-web";

const INPUT_SIZE = 256;
const KEYPOINT_THRESHOLD = 0.3;
const KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
];
const KEYPOINT_INDEX = Object.fromEntries(KEYPOINT_NAMES.map((name, index) => [name, index]));
const CAMERA_ASSET_BASE = new URL(import.meta.env.BASE_URL, window.location.origin);
const cameraAssetUrl = (path) => new URL(path, CAMERA_ASSET_BASE).href;

function mapKeypointsToSource(keypoints, width, height) {
    const mapped = keypoints.map((point) => ({ ...point }));
    if (width > height) {
        const scale = height / width;
        const verticalPadding = (1 - scale) / 2;
        mapped.forEach((point) => {
            point.y = (point.y - verticalPadding) / scale;
        });
    } else if (height > width) {
        const scale = width / height;
        const horizontalPadding = (1 - scale) / 2;
        mapped.forEach((point) => {
            point.x = (point.x - horizontalPadding) / scale;
        });
    }
    return mapped;
}

function detectPerson(keypoints) {
    const visible = keypoints.map((point) => point.score >= KEYPOINT_THRESHOLD);
    const shoulderVisible = visible[5] || visible[6];
    const hipVisible = visible[11] || visible[12];
    return visible.filter(Boolean).length >= 6 && shoulderVisible && hipVisible;
}

function extractFeatures(keypoints, metadata) {
    if (metadata.feature_count !== 51 || metadata.feature_order.length !== 51) {
        throw new Error("The browser classifier metadata does not match the expected 51 features.");
    }

    return Float32Array.from(metadata.feature_order.map((featureName) => {
        const separator = featureName.lastIndexOf("_");
        const keypointName = featureName.slice(0, separator);
        const component = featureName.slice(separator + 1);
        const index = KEYPOINT_INDEX[keypointName];
        const point = keypoints[index];
        if (index === undefined || !point || !["x", "y", "score"].includes(component)) {
            throw new Error(`Unsupported classifier feature in model metadata: ${featureName}.`);
        }
        return point[component];
    }));
}

function normalizePoseName(value) {
    const normalized = String(value || "").trim().toLowerCase().replaceAll(" ", "_").replaceAll("-", "_");
    if (normalized === "shoulder_stand") {
        return "shoudler_stand";
    }
    if (normalized === "triangle") {
        return "traingle";
    }
    return normalized;
}

function jointAngle(first, middle, last) {
    const firstVector = [first.x - middle.x, first.y - middle.y];
    const secondVector = [last.x - middle.x, last.y - middle.y];
    const denominator = Math.hypot(...firstVector) * Math.hypot(...secondVector);
    if (denominator <= 1e-6) {
        return null;
    }
    const cosine = Math.max(-1, Math.min(1,
        (firstVector[0] * secondVector[0] + firstVector[1] * secondVector[1]) / denominator
    ));
    return Math.acos(cosine) * (180 / Math.PI);
}

function pointForAngle(keypoints, name, width, height, useSourceCoordinates) {
    const point = keypoints[KEYPOINT_INDEX[name]];
    if (!point || point.score < KEYPOINT_THRESHOLD) {
        return null;
    }
    if (useSourceCoordinates) {
        return { x: point.x * width, y: point.y * height };
    }
    return { x: point.x, y: point.y };
}

function angleFor(keypoints, names, width, height, useSourceCoordinates) {
    const points = names.map((name) =>
        pointForAngle(keypoints, name, width, height, useSourceCoordinates)
    );
    return points.some((point) => point === null)
        ? null
        : jointAngle(points[0], points[1], points[2]);
}

function generateFeedback(keypoints, sourceKeypoints, prediction, expectedPose, width, height) {
    const cues = [];
    const visible = (name) =>
        keypoints[KEYPOINT_INDEX[name]]?.score >= KEYPOINT_THRESHOLD;

    if (visible("left_shoulder") && visible("right_shoulder")) {
        const left = keypoints[KEYPOINT_INDEX.left_shoulder];
        const right = keypoints[KEYPOINT_INDEX.right_shoulder];
        if (Math.abs(left.y - right.y) > 0.05) {
            cues.push("Level your shoulders.");
        }
    }

    for (const side of ["left", "right"]) {
        const angle = angleFor(
            keypoints,
            [`${side}_shoulder`, `${side}_elbow`, `${side}_wrist`],
            width,
            height,
            false
        );
        if (angle !== null && angle < 160) {
            cues.push(`Straighten your ${side} arm.`);
        }
    }

    if (visible("left_hip") && visible("right_hip")) {
        const left = keypoints[KEYPOINT_INDEX.left_hip];
        const right = keypoints[KEYPOINT_INDEX.right_hip];
        if (Math.abs(left.y - right.y) > 0.06) {
            cues.push("Level your hips.");
        }
    }

    const normalizedPrediction = normalizePoseName(prediction);
    const normalizedExpected = normalizePoseName(expectedPose);
    if (normalizedPrediction === "no_pose") {
        cues.unshift("Find your center and settle into a balanced stance.");
    } else if (normalizedPrediction === "chair" || normalizedPrediction === "warrior") {
        const kneeAngles = ["left", "right"].map((side) =>
            angleFor(
                sourceKeypoints,
                [`${side}_hip`, `${side}_knee`, `${side}_ankle`],
                width,
                height,
                true
            )
        ).filter((angle) => angle !== null);
        if (kneeAngles.length && Math.min(...kneeAngles) > 145) {
            cues.unshift("Bend your knees a little more while keeping them aligned with your feet.");
        } else if (kneeAngles.length === 2 && Math.abs(kneeAngles[0] - kneeAngles[1]) > 25) {
            cues.unshift("Balance the bend between both legs for a steadier stance.");
        }
    } else if (normalizedPrediction === "tree") {
        const kneeAngles = ["left", "right"].map((side) =>
            angleFor(
                sourceKeypoints,
                [`${side}_hip`, `${side}_knee`, `${side}_ankle`],
                width,
                height,
                true
            )
        );
        if (kneeAngles.some((angle) => angle !== null && angle >= 145 && angle < 165)) {
            cues.unshift("Lengthen your standing leg without locking the knee.");
        }
    } else if (normalizedPrediction === "cobra") {
        const elbowAngles = ["left", "right"].map((side) =>
            angleFor(
                sourceKeypoints,
                [`${side}_shoulder`, `${side}_elbow`, `${side}_wrist`],
                width,
                height,
                true
            )
        );
        if (elbowAngles.some((angle) => angle !== null && angle < 145)) {
            cues.unshift("Ease the backbend by softening your elbows and relaxing your shoulders.");
        }
    }

    if (expectedPose && normalizedPrediction && normalizedPrediction !== normalizedExpected) {
        const expectedNames = {
            chair: "Chair",
            cobra: "Cobra",
            dog: "Downward Dog",
            warrior: "Warrior",
            tree: "Tree",
            traingle: "Triangle",
            shoudler_stand: "Shoulder Stand"
        };
        cues.unshift(`Move toward your selected ${expectedNames[normalizedExpected] || "yoga"} pose.`);
    }

    if (!cues.length) {
        const displayNames = {
            chair: "Chair",
            cobra: "Cobra",
            dog: "Downward Dog",
            warrior: "Warrior",
            tree: "Tree",
            traingle: "Triangle",
            shoudler_stand: "Shoulder Stand"
        };
        cues.push(`Your ${displayNames[normalizedPrediction] || "yoga"} landmarks are visible; keep breathing and move within a comfortable range.`);
    }
    return cues.slice(0, 3);
}

export class BrowserPoseAnalyzer {
    static async create() {
        await tf.setBackend("cpu");
        await tf.ready();
        const wasmBase = cameraAssetUrl("wasm/");
        tflite.setWasmPath(wasmBase);
        ort.env.wasm.wasmPaths = {
            mjs: cameraAssetUrl("wasm/ort-wasm-simd-threaded.mjs"),
            wasm: cameraAssetUrl("wasm/ort-wasm-simd-threaded.wasm")
        };
        ort.env.wasm.numThreads = 1;

        const [poseModel, classifier, metadataResponse] = await Promise.all([
            tflite.loadTFLiteModel(cameraAssetUrl("models/movenet_thunder.tflite"), {
                numThreads: 1
            }),
            ort.InferenceSession.create(cameraAssetUrl("models/pose_cls_xgboost.onnx"), {
                executionProviders: ["wasm"]
            }),
            fetch(cameraAssetUrl("models/pose_cls_xgboost.json"))
        ]);
        if (!metadataResponse.ok) {
            throw new Error(`Classifier metadata request failed (${metadataResponse.status}).`);
        }
        const metadata = await metadataResponse.json();
        if (
            metadata.format_version !== 1
            || metadata.feature_count !== 51
            || metadata.feature_order?.length !== 51
            || metadata.use_scores !== true
            || metadata.class_labels?.length !== 8
        ) {
            throw new Error("The browser classifier metadata is incomplete or incompatible.");
        }
        if (poseModel.inputs.length !== 1
            || poseModel.inputs[0].shape.join(",") !== "1,256,256,3"
            || poseModel.outputs.length !== 1
            || poseModel.outputs[0].shape.join(",") !== "1,1,17,3") {
            throw new Error("The MoveNet TFLite tensor shapes do not match the expected model.");
        }
        return new BrowserPoseAnalyzer(poseModel, classifier, metadata);
    }

    constructor(poseModel, classifier, metadata) {
        this.poseModel = poseModel;
        this.classifier = classifier;
        this.metadata = metadata;
        this.inputCanvas = document.createElement("canvas");
        this.inputCanvas.width = INPUT_SIZE;
        this.inputCanvas.height = INPUT_SIZE;
        this.inputContext = this.inputCanvas.getContext("2d", { willReadFrequently: true });
        if (!this.inputContext) {
            throw new Error("Could not create the browser inference canvas.");
        }
    }

    async analyze(video, expectedPose) {
        const width = video.videoWidth;
        const height = video.videoHeight;
        if (!width || !height) {
            throw new Error("The camera frame is not ready for analysis.");
        }

        this.inputContext.fillStyle = "#000";
        this.inputContext.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
        this.inputContext.imageSmoothingEnabled = true;
        this.inputContext.imageSmoothingQuality = "high";
        const scale = Math.min(INPUT_SIZE / width, INPUT_SIZE / height);
        const resizedWidth = width * scale;
        const resizedHeight = height * scale;
        this.inputContext.drawImage(
            video,
            (INPUT_SIZE - resizedWidth) / 2,
            (INPUT_SIZE - resizedHeight) / 2,
            resizedWidth,
            resizedHeight
        );

        const rgba = this.inputContext.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
        const rgb = new Int32Array(INPUT_SIZE * INPUT_SIZE * 3);
        for (let sourceIndex = 0, targetIndex = 0; sourceIndex < rgba.length; sourceIndex += 4) {
            rgb[targetIndex++] = rgba[sourceIndex];
            rgb[targetIndex++] = rgba[sourceIndex + 1];
            rgb[targetIndex++] = rgba[sourceIndex + 2];
        }

        const input = tf.tensor4d(rgb, [1, INPUT_SIZE, INPUT_SIZE, 3], "int32");
        let output;
        let rawKeypoints;
        try {
            output = this.poseModel.predict(input);
            rawKeypoints = Array.from(await output.data());
        } finally {
            input.dispose();
            output?.dispose();
        }

        const keypoints = KEYPOINT_NAMES.map((name, index) => ({
            name,
            y: rawKeypoints[index * 3],
            x: rawKeypoints[index * 3 + 1],
            score: rawKeypoints[index * 3 + 2]
        }));
        const sourceKeypoints = mapKeypointsToSource(keypoints, width, height);
        const visibleKeypoints = keypoints.filter((point) => point.score >= KEYPOINT_THRESHOLD).length;
        const keypointConfidence = keypoints.reduce((sum, point) => sum + point.score, 0) / keypoints.length;
        const personDetected = detectPerson(keypoints);
        const poseDetectionStatus = !personDetected
            ? "Not detected"
            : visibleKeypoints >= 12 && keypointConfidence >= 0.5
                ? "Good"
                : "Partial";
        const sourcePointsForOverlay = sourceKeypoints.map((point) => ({
            name: point.name,
            x: point.x,
            y: point.y,
            score: point.score
        }));

        if (!personDetected) {
            return {
                pose: "no_pose",
                confidence: null,
                confidence_type: null,
                person_detected: false,
                pose_detection_status: poseDetectionStatus,
                keypoint_confidence: keypointConfidence,
                visible_keypoints: visibleKeypoints,
                keypoints: sourcePointsForOverlay,
                feedback: ["Please move into the camera frame so your body can be detected clearly."]
            };
        }

        const features = extractFeatures(keypoints, this.metadata);
        const classifierInput = new ort.Tensor("float32", features, [1, features.length]);
        const outputs = await this.classifier.run({
            [this.metadata.input_name]: classifierInput
        });
        const classIndex = Number(outputs[this.metadata.outputs.label].data[0]);
        const probabilities = outputs[this.metadata.outputs.probabilities].data;
        const classPosition = this.metadata.class_indices.indexOf(classIndex);
        if (classPosition < 0 || probabilities.length !== this.metadata.class_labels.length) {
            throw new Error("The browser classifier returned an unknown class or probability shape.");
        }

        const pose = this.metadata.class_labels[classPosition];
        const confidence = Number(probabilities[classPosition]);
        const feedback = generateFeedback(
            keypoints,
            sourceKeypoints,
            pose,
            expectedPose,
            width,
            height
        );
        return {
            pose,
            confidence,
            confidence_type: "predict_proba",
            person_detected: true,
            pose_detection_status: poseDetectionStatus,
            keypoint_confidence: keypointConfidence,
            visible_keypoints: visibleKeypoints,
            keypoints: sourcePointsForOverlay,
            feedback
        };
    }
}
