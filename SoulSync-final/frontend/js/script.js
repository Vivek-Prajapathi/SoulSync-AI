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
const targetPose = document.getElementById("targetPose");
const selectedPose = document.getElementById("selectedPose");
const coachMessage = document.getElementById("coachMessage");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const camera = document.getElementById("camera");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const cameraStatus = document.getElementById("cameraStatus");
const cameraOverlay = document.querySelector(".camera-overlay-label");
const cameraMessage = document.getElementById("cameraMessage");
const holdTime = document.getElementById("holdTime");

let currentPose = null;
let cameraStream = null;
let sessionStartedAt = 0;
let sessionTimer = null;

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

    targetPose.textContent = pose.name;
    selectedPose.textContent = pose.sanskrit;
    coachMessage.textContent = cameraStream
        ? `Your ${pose.name} practice is selected. Move gently and listen to your body.`
        : `A lovely choice. Turn on your camera when you're ready to practice ${pose.name}.`;
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

async function startCamera() {
    cameraMessage.textContent = "";
    cameraMessage.classList.remove("is-error");

    if (!currentPose) {
        cameraMessage.textContent = "Choose a pose first, then turn on your camera.";
        return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showCameraError({ name: "UnsupportedError" });
        return;
    }

    startBtn.disabled = true;
    startBtn.textContent = "Connecting…";
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user" },
            audio: false
        });
        camera.srcObject = cameraStream;
        camera.style.display = "block";
        cameraPlaceholder.style.display = "none";
        stopBtn.disabled = false;
        setCameraStatus(true);
        sessionStartedAt = Date.now();
        formatElapsedTime();
        sessionTimer = window.setInterval(formatElapsedTime, 1000);
        cameraMessage.textContent = "Camera preview is on. Pose analysis is not connected to the browser yet.";
        coachMessage.textContent = `Your ${poses[currentPose].name} practice is ready. Settle into a comfortable position and move with care.`;
    } catch (error) {
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
    camera.srcObject = null;
    camera.style.display = "none";
    cameraPlaceholder.style.display = "flex";
    stopBtn.disabled = true;
    holdTime.textContent = "0:00";
    setCameraStatus(false);
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
