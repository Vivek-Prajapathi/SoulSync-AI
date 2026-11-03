import { BrowserPoseAnalyzer } from "./browser-inference.js";

const poses = {
    cobra: {
        name: "Cobra",
        sanskrit: "Bhujangasana"
    },
    chair: {
        name: "Chair",
        sanskrit: "Utkatasana"
    },
    warrior: {
        name: "Warrior",
        sanskrit: "Virabhadrasana"
    },
    dog: {
        name: "Downward dog",
        sanskrit: "Adho Mukha Svanasana"
    },
    tree: {
        name: "Tree",
        sanskrit: "Vrksasana"
    },
    triangle: {
        name: "Triangle",
        sanskrit: "Trikonasana"
    },
    "shoulder-stand": {
        name: "Shoulder stand",
        sanskrit: "Sarvangasana"
    }
};

const poseGrid = document.getElementById("poseGrid");
const detectedPose = document.getElementById("detectedPose");
const selectedPose = document.getElementById("selectedPose");
const coachMessage = document.getElementById("coachMessage");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const camera = document.getElementById("camera");
const skeletonOverlay = document.getElementById("skeletonOverlay");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const cameraStatus = document.getElementById("cameraStatus");
const cameraOverlay = document.querySelector(".camera-overlay-label");
const cameraMessage = document.getElementById("cameraMessage");
const holdTime = document.getElementById("holdTime");
const predictionConfidence = document.getElementById("predictionConfidence");
const keypointConfidence = document.getElementById("keypointConfidence");
const poseDetectionStatus = document.getElementById("poseDetectionStatus");
const poseAlternatives = document.getElementById("poseAlternatives");

let currentPose = null;
let cameraStream = null;
let sessionStartedAt = 0;
let sessionTimer = null;
let analysisFrame = null;
let analysisInFlight = false;
let lastAnalysisTime = 0;
let browserAnalyzer = null;
let analyzerPromise = null;

function selectPose(poseId) {
    const pose = poses[poseId];
    if (!pose) {
        return;
    }

    currentPose = poseId;
    document.querySelectorAll(".pose-card").forEach((card) => {
        const isSelected = card.dataset.pose === poseId;
        card.classList.toggle("selected", isSelected);
        card.setAttribute("aria-pressed", String(isSelected));
    });

    detectedPose.textContent = cameraStream ? "Waiting for analysis…" : "Camera off";
    selectedPose.textContent = pose.sanskrit;
    predictionConfidence.textContent = "--";
    poseAlternatives.textContent = "";
    coachMessage.textContent = cameraStream
        ? `Your ${pose.name} practice is selected. Move gently and listen to your body.`
        : `A lovely choice. Turn on your camera when you're ready to practice ${pose.name}.`;

    if (cameraStream) {
        analyzeCurrentFrame();
    }
}

function createPoseCard(poseId, pose, index, selectedId, onSelect) {
    const h = React.createElement;
    return h(
        "button",
        {
            key: poseId,
            className: `pose-card${selectedId === poseId ? " selected" : ""}`,
            type: "button",
            "data-pose": poseId,
            "aria-pressed": String(selectedId === poseId),
            onClick: () => onSelect(poseId)
        },
        h("span", { className: "pose-index", "aria-hidden": "true" }, String(index + 1).padStart(2, "0")),
        h("span", { className: "pose-card-copy" },
            h("strong", null, pose.name),
            h("small", null, pose.sanskrit)
        ),
        h("span", { className: "pose-arrow", "aria-hidden": "true" }, "↗")
    );
}

function PosePicker() {
    const [selectedId, setSelectedId] = React.useState(null);
    const handleSelect = (poseId) => {
        setSelectedId(poseId);
        selectPose(poseId);
    };

    return React.createElement(
        React.Fragment,
        null,
        Object.entries(poses).map(([poseId, pose], index) =>
            createPoseCard(poseId, pose, index, selectedId, handleSelect)
        )
    );
}

if (poseGrid && window.React && window.ReactDOM) {
    ReactDOM.createRoot(poseGrid).render(React.createElement(PosePicker));
} else if (poseGrid) {
    poseGrid.querySelectorAll(".pose-card").forEach((card) => {
        card.addEventListener("click", () => selectPose(card.dataset.pose));
    });
}

function formatElapsedTime() {
    const elapsedSeconds = Math.floor((Date.now() - sessionStartedAt) / 1000);
    const minutes = Math.floor(elapsedSeconds / 60).toString();
    const seconds = (elapsedSeconds % 60).toString().padStart(2, "0");
    holdTime.textContent = `${minutes}:${seconds}`;
}

function setCameraStatus(isOn) {
    cameraStatus.classList.toggle("is-on", isOn);
    cameraStatus.lastChild.textContent = isOn ? " CAMERA ON" : " CAMERA OFF";
    cameraOverlay.textContent = isOn ? "PREVIEW ON" : "PREVIEW OFF";
    cameraOverlay.classList.toggle("is-on", isOn);
}

function showCameraError(error) {
    cameraMessage.classList.add("is-error");
    if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
        cameraMessage.textContent = "Camera permission was denied. Allow camera access in your browser settings, then try again.";
    } else if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
        cameraMessage.textContent = "No camera was found. Connect a camera and try again.";
    } else if (error.name === "NotReadableError" || error.name === "TrackStartError") {
        cameraMessage.textContent = "Your camera could not be started. Close other apps using it and try again.";
    } else {
        cameraMessage.textContent = "Camera access is unavailable. Use a secure connection and check your browser's camera settings.";
    }
}

const skeletonEdges = [
    [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6], [5, 6],
    [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
];

function confidenceColor(score) {
    if (score >= 0.65) {
        return "#38e078";
    }
    if (score >= 0.4) {
        return "#ffd447";
    }
    return "#ff5b50";
}

function drawSkeleton(keypoints) {
    const context = skeletonOverlay.getContext("2d");
    const bounds = skeletonOverlay.getBoundingClientRect();
    const pixelRatio = window.devicePixelRatio || 1;
    const width = bounds.width;
    const height = bounds.height;

    skeletonOverlay.width = Math.round(width * pixelRatio);
    skeletonOverlay.height = Math.round(height * pixelRatio);
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, width, height);

    if (!Array.isArray(keypoints) || keypoints.length !== 17 || !camera.videoWidth || !camera.videoHeight) {
        return;
    }

    const coverScale = Math.max(width / camera.videoWidth, height / camera.videoHeight);
    const renderedWidth = camera.videoWidth * coverScale;
    const renderedHeight = camera.videoHeight * coverScale;
    const offsetX = (width - renderedWidth) / 2;
    const offsetY = (height - renderedHeight) / 2;
    const point = (index) => ({
        x: keypoints[index].x * renderedWidth + offsetX,
        y: keypoints[index].y * renderedHeight + offsetY,
        score: keypoints[index].score
    });

    context.lineCap = "round";
    context.lineJoin = "round";
    context.shadowBlur = 7;
    for (const [startIndex, endIndex] of skeletonEdges) {
        const start = point(startIndex);
        const end = point(endIndex);
        if (start.score < 0.25 || end.score < 0.25) {
            continue;
        }

        context.beginPath();
        context.strokeStyle = confidenceColor((start.score + end.score) / 2);
        context.lineWidth = 3.5;
        context.shadowColor = context.strokeStyle;
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.stroke();
    }

    context.shadowBlur = 9;
    keypoints.forEach((keypoint, index) => {
        if (keypoint.score < 0.25) {
            return;
        }

        const joint = point(index);
        const radius = keypoint.score >= 0.4 ? 6 : 5;
        context.beginPath();
        context.fillStyle = confidenceColor(keypoint.score);
        context.shadowColor = context.fillStyle;
        context.arc(joint.x, joint.y, radius, 0, Math.PI * 2);
        context.fill();
        context.beginPath();
        context.fillStyle = "#ffffff";
        context.shadowBlur = 0;
        context.arc(joint.x, joint.y, 2, 0, Math.PI * 2);
        context.fill();
        context.shadowBlur = 9;
    });
    context.shadowBlur = 0;
}

function canonicalPose(label) {
    const normalized = String(label || "").toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
    if (normalized === "shoudler_stand" || normalized === "shoulder_stand") {
        return "shoulder-stand";
    }
    if (normalized === "traingle" || normalized === "triangle") {
        return "triangle";
    }
    return normalized;
}

function poseName(label) {
    if (canonicalPose(label) === "no_pose") {
        return "No pose";
    }
    const selected = poses[canonicalPose(label)];
    return selected ? selected.name : String(label || "No pose").replaceAll("_", " ");
}

function setNoDetectionState() {
    drawSkeleton([]);
    detectedPose.textContent = "No pose detected";
    predictionConfidence.textContent = "--";
    poseAlternatives.textContent = "";
    poseDetectionStatus.textContent = "Not detected";
    cameraMessage.textContent = "Please move into the camera frame so your body can be detected clearly.";
    coachMessage.textContent = cameraMessage.textContent;
}

function loadBrowserAnalyzer() {
    if (browserAnalyzer) {
        return Promise.resolve(browserAnalyzer);
    }
    if (!analyzerPromise) {
        analyzerPromise = BrowserPoseAnalyzer.create()
            .then((analyzer) => {
                browserAnalyzer = analyzer;
                return analyzer;
            })
            .catch((error) => {
                analyzerPromise = null;
                throw error;
            });
    }
    return analyzerPromise;
}

async function analyzeCurrentFrame() {
    if (!cameraStream || !camera.videoWidth || !camera.videoHeight || analysisInFlight || !browserAnalyzer) {
        return;
    }

    analysisInFlight = true;

    try {
        const result = await browserAnalyzer.analyze(camera, currentPose);
        if (!cameraStream) {
            return;
        }

        drawSkeleton(result.keypoints);
        const pointConfidence = result.keypoint_confidence === null ? Number.NaN : Number(result.keypoint_confidence);
        keypointConfidence.textContent = Number.isFinite(pointConfidence)
            ? `${(pointConfidence * 100).toFixed(1)}% · ${result.visible_keypoints}/17 visible`
            : "--";
        poseDetectionStatus.textContent = result.pose_detection_status || "Unknown";
        cameraMessage.classList.remove("is-error");

        if (!result.person_detected) {
            setNoDetectionState();
            return;
        }

        const predictedName = poseName(result.pose);
        const classifierConfidence = result.confidence === null ? Number.NaN : Number(result.confidence);
        detectedPose.textContent = predictedName;
        predictionConfidence.textContent = Number.isFinite(classifierConfidence)
            ? `${(classifierConfidence * 100).toFixed(1)}%`
            : "Unavailable";
        poseAlternatives.textContent = Array.isArray(result.top_predictions)
            ? `Top model outputs: ${result.top_predictions
                .map(({ pose, confidence }) => `${poseName(pose)} ${(confidence * 100).toFixed(1)}%`)
                .join(" · ")}`
            : "";

        const target = poses[currentPose]?.name;
        const matchesTarget = currentPose && canonicalPose(result.pose) === canonicalPose(currentPose);
        cameraMessage.textContent = !target
            ? `${predictedName} detected by on-device MoveNet and XGBoost.`
            : matchesTarget
                ? `${predictedName} detected · target pose matched`
                : `${predictedName} detected · selected target: ${target}`;
        coachMessage.textContent = Array.isArray(result.feedback) && result.feedback.length
            ? result.feedback[0]
            : "Pose detected. Keep breathing and move within a comfortable range.";
    } catch (error) {
        if (cameraStream) {
            drawSkeleton([]);
            cameraMessage.classList.add("is-error");
            cameraMessage.textContent = `On-device pose analysis unavailable: ${error.message}`;
            poseDetectionStatus.textContent = "AI unavailable";
            predictionConfidence.textContent = "--";
            poseAlternatives.textContent = "";
            coachMessage.textContent = "Check your connection while the browser downloads the models, then try again.";
        }
    } finally {
        analysisInFlight = false;
    }
}

function analysisLoop(timestamp) {
    if (!cameraStream || document.hidden || !browserAnalyzer) {
        return;
    }

    if (timestamp - lastAnalysisTime >= 250 && !analysisInFlight) {
        lastAnalysisTime = timestamp;
        analyzeCurrentFrame();
    }
    analysisFrame = window.requestAnimationFrame(analysisLoop);
}

function startAnalysisLoop() {
    if (analysisFrame !== null) {
        window.cancelAnimationFrame(analysisFrame);
    }
    lastAnalysisTime = performance.now() - 900;
    analysisFrame = window.requestAnimationFrame(analysisLoop);
}

async function startCamera() {
    cameraMessage.textContent = "";
    cameraMessage.classList.remove("is-error");

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showCameraError({ name: "UnsupportedError" });
        return;
    }

    startBtn.disabled = true;
    startBtn.textContent = "Connecting…";
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false
        });
        camera.srcObject = cameraStream;
        camera.style.display = "block";
        cameraPlaceholder.style.display = "none";
        await camera.play();
        stopBtn.disabled = false;
        setCameraStatus(true);
        sessionStartedAt = Date.now();
        formatElapsedTime();
        sessionTimer = window.setInterval(formatElapsedTime, 1000);
        detectedPose.textContent = "Loading on-device AI…";
        poseDetectionStatus.textContent = "Loading models";
        poseAlternatives.textContent = "";
        cameraMessage.textContent = "Live camera is on. Loading MoveNet and the trained pose classifier in your browser.";
        coachMessage.textContent = "Your camera frames stay on this device while the models load.";
        loadBrowserAnalyzer()
            .then(() => {
                if (!cameraStream) {
                    return;
                }
                detectedPose.textContent = "Looking for body…";
                poseDetectionStatus.textContent = "Looking for body";
                cameraMessage.classList.remove("is-error");
                cameraMessage.textContent = "On-device AI is ready. Move into view for pose detection.";
                startAnalysisLoop();
            })
            .catch((error) => {
                if (!cameraStream) {
                    return;
                }
                cameraMessage.classList.add("is-error");
                cameraMessage.textContent = `Could not load on-device AI models: ${error.message}`;
                poseDetectionStatus.textContent = "AI unavailable";
                coachMessage.textContent = "The camera preview remains on. Check your connection, then turn the camera off and try again.";
            });
    } catch (error) {
        if (cameraStream) {
            cameraStream.getTracks().forEach((track) => track.stop());
            cameraStream = null;
        }
        camera.srcObject = null;
        camera.style.display = "none";
        cameraPlaceholder.style.display = "flex";
        showCameraError(error);
    } finally {
        startBtn.disabled = false;
        startBtn.innerHTML = 'Turn on camera <span aria-hidden="true">→</span>';
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
        cameraStream = null;
    }
    if (sessionTimer) {
        window.clearInterval(sessionTimer);
        sessionTimer = null;
    }
    if (analysisFrame !== null) {
        window.cancelAnimationFrame(analysisFrame);
        analysisFrame = null;
    }
    camera.srcObject = null;
    camera.style.display = "none";
    cameraPlaceholder.style.display = "flex";
    drawSkeleton([]);
    stopBtn.disabled = true;
    holdTime.textContent = "0:00";
    setCameraStatus(false);
    detectedPose.textContent = currentPose ? "Camera off" : "Choose a pose";
    predictionConfidence.textContent = "--";
    poseAlternatives.textContent = "";
    keypointConfidence.textContent = "--";
    poseDetectionStatus.textContent = "Camera off";
    cameraMessage.textContent = "Camera is off. Take a moment, then start again whenever you're ready.";
    cameraMessage.classList.remove("is-error");
    coachMessage.textContent = currentPose
        ? `Your ${poses[currentPose].name} is still selected. Start again whenever you're ready.`
        : "Take a breath. Choose a pose to set an intention for your practice.";
}

if (startBtn && stopBtn && camera) {
    startBtn.addEventListener("click", startCamera);
    stopBtn.addEventListener("click", stopCamera);
}

document.addEventListener("visibilitychange", () => {
    if (document.hidden && analysisFrame !== null) {
        window.cancelAnimationFrame(analysisFrame);
        analysisFrame = null;
    } else if (!document.hidden && cameraStream && analysisFrame === null) {
        startAnalysisLoop();
    }
});

window.addEventListener("pagehide", stopCamera);
