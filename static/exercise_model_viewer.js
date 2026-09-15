import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const VALID_FORMATS = new Set(['fbx', 'gltf', 'glb']);

class ExerciseModelViewer {
    constructor(root) {
        this.root = root;
        this.canvasHost = root.querySelector('[data-viewer-canvas]');
        this.statusEl = root.querySelector('[data-viewer-status]');
        this.playButton = root.querySelector('[data-viewer-play]');
        this.resetButton = root.querySelector('[data-viewer-reset]');
        this.speedSelect = root.querySelector('[data-viewer-speed]');

        this.modelUrl = root.dataset.modelUrl || '';
        this.modelFormat = (root.dataset.modelFormat || 'none').toLowerCase();
        this.animationKey = root.dataset.animationKey || root.dataset.exerciseSlug || '';
        this.introModelUrl = root.dataset.introModelUrl || '';
        this.introModelFormat = (root.dataset.introModelFormat || 'none').toLowerCase();
        this.introAnimationKey = root.dataset.introAnimationKey || '';
        this.exerciseSlug = root.dataset.exerciseSlug || '';

        this.playing = true;
        this.speed = 1;
        this.clock = new THREE.Clock();
        this.mixer = null;
        this.activeAction = null;
        this.avatar = null;
        this.activeStage = '';
        this.mainAsset = null;
        this.introAsset = null;
        this.mixerFinishedHandler = null;

        this.initScene();
        this.bindControls();
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
        this.loadContent();
    }

    initScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0b1425);

        this.camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
        this.camera.position.set(3.4, 2.2, 5.2);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.8));
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.canvasHost.replaceChildren(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.target.set(0, 1.05, 0);
        this.controls.minDistance = 2.4;
        this.controls.maxDistance = 9;
        this.controls.maxPolarAngle = Math.PI * 0.49;
        this.controls.saveState();

        this.scene.add(new THREE.HemisphereLight(0xcfe5ff, 0x263040, 2.0));

        const key = new THREE.DirectionalLight(0xffffff, 3.0);
        key.position.set(3, 6, 4);
        key.castShadow = true;
        key.shadow.mapSize.set(1024, 1024);
        this.scene.add(key);

        const fill = new THREE.DirectionalLight(0x6aa5ff, 1.15);
        fill.position.set(-4, 2, -2);
        this.scene.add(fill);

        const rim = new THREE.DirectionalLight(0xc6ddff, 1.0);
        rim.position.set(0, 4, -5);
        this.scene.add(rim);

        const floor = new THREE.Mesh(
            new THREE.CircleGeometry(3.6, 64),
            new THREE.MeshStandardMaterial({ color: 0x172238, roughness: 0.92, metalness: 0.02 })
        );
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        this.scene.add(floor);

        const grid = new THREE.GridHelper(7, 14, 0x42638c, 0x263650);
        grid.position.y = 0.006;
        this.scene.add(grid);

        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(this.canvasHost);
        this.resize();
    }

    bindControls() {
        this.playButton?.addEventListener('click', () => {
            if (!this.activeAction) return;
            this.playing = !this.playing;
            this.playButton.textContent = this.playing ? 'Tạm dừng' : 'Tiếp tục';
            this.activeAction.paused = !this.playing;
        });

        this.resetButton?.addEventListener('click', () => {
            this.playing = true;
            if (this.playButton) this.playButton.textContent = 'Tạm dừng';
            this.controls.reset();
            this.startSequence();
        });

        this.speedSelect?.addEventListener('change', () => {
            this.speed = clamp(Number(this.speedSelect.value || 1), 0.25, 2.5);
            if (this.mixer) this.mixer.timeScale = this.speed;
        });
    }

    setStatus(message, kind = 'neutral') {
        if (!this.statusEl) return;
        const clean = String(message || '').trim();
        this.statusEl.textContent = clean;
        this.statusEl.dataset.state = kind;
        this.statusEl.hidden = !clean;
    }

    setControlsEnabled(enabled) {
        if (this.playButton) this.playButton.disabled = !enabled;
        if (this.resetButton) this.resetButton.disabled = !enabled;
        if (this.speedSelect) this.speedSelect.disabled = !enabled;
    }

    isAssetConfigured(url, format) {
        return Boolean(url) && VALID_FORMATS.has(String(format || '').toLowerCase());
    }

    async loadAsset(url, format) {
        const normalizedFormat = String(format || '').toLowerCase();
        if (normalizedFormat === 'fbx') {
            const model = await new FBXLoader().loadAsync(url);
            return { model, clips: model.animations || [] };
        }
        const gltf = await new GLTFLoader().loadAsync(url);
        return { model: gltf.scene, clips: gltf.animations || [] };
    }

    prepareModel(model) {
        model.traverse((node) => {
            if (node.isMesh) {
                node.castShadow = true;
                node.receiveShadow = true;
                const materials = Array.isArray(node.material) ? node.material : [node.material];
                materials.filter(Boolean).forEach((material) => {
                    if (material.map) material.map.colorSpace = THREE.SRGBColorSpace;
                });
            }
        });
        return model;
    }

    async loadContent() {
        if (!this.isAssetConfigured(this.modelUrl, this.modelFormat)) {
            this.setControlsEnabled(false);
            // Keep the user workout screen clean. Missing-asset guidance belongs
            // in Admin, not as a persistent yellow comment under the 3D box.
            // Khi thiếu asset thật, hệ thống không dùng mô hình giả thay thế.
            this.setStatus('', 'neutral');
            return;
        }

        this.setControlsEnabled(false);
        this.setStatus('Đang tải mô hình Mixamo/Blender...', 'loading');
        try {
            this.mainAsset = await this.loadAsset(this.modelUrl, this.modelFormat);
            this.mainAsset.model = this.prepareModel(this.mainAsset.model);
            if (!this.mainAsset.clips.length) {
                throw new Error('Mô hình chính không có animation clip.');
            }

            if (this.isAssetConfigured(this.introModelUrl, this.introModelFormat)) {
                try {
                    this.introAsset = await this.loadAsset(this.introModelUrl, this.introModelFormat);
                    this.introAsset.model = this.prepareModel(this.introAsset.model);
                    if (!this.introAsset.clips.length) {
                        this.introAsset = null;
                        // User screen stays clean; Admin is the place to fix an invalid intro asset.
                    }
                } catch (introError) {
                    console.warn('Không tải được animation mở đầu, tiếp tục bằng animation chính.', introError);
                    this.introAsset = null;
                }
            }

            this.setControlsEnabled(true);
            this.startSequence();
        } catch (error) {
            console.error('Không tải được mô hình 3D thật', error);
            this.setControlsEnabled(false);
            this.setStatus(
                'Không tải được FBX/GLB hoặc file không có animation. Hãy kiểm tra asset Mixamo, texture, rig và animation.',
                'error'
            );
        }
    }

    selectClip(clips, key) {
        const wanted = String(key || '').trim().toLowerCase();
        if (wanted) {
            const exact = clips.find((item) => String(item.name || '').trim().toLowerCase() === wanted);
            if (exact) return exact;
            const contains = clips.find((item) => String(item.name || '').trim().toLowerCase().includes(wanted));
            if (contains) return contains;
        }
        return clips[0];
    }

    isArmTrackForSide(trackName, side) {
        const name = String(trackName || '').toLowerCase();
        const sideToken = String(side || '').toLowerCase();
        if (!sideToken || !name.includes(sideToken)) return false;
        return ['shoulder', 'arm', 'forearm', 'hand', 'thumb', 'index', 'middle', 'ring', 'pinky']
            .some((token) => name.includes(token));
    }

    freezeTrackAtFirstPose(track) {
        const frozen = track.clone();
        const valueSize = Math.max(1, frozen.getValueSize());
        const first = Array.from(frozen.values.slice(0, valueSize));
        for (let offset = 0; offset < frozen.values.length; offset += valueSize) {
            for (let index = 0; index < valueSize; index += 1) {
                frozen.values[offset + index] = first[index];
            }
        }
        return frozen;
    }

    prepareClipForExercise(clip) {
        if (!clip || !['curl-left', 'curl-right'].includes(this.exerciseSlug)) return clip;
        // Mixamo's stock "Bicep Curl" moves both arms. FitMotion keeps the
        // selected arm animated and freezes only the opposite arm at frame 1,
        // so the same stock Mixamo clip can represent curl-left/curl-right.
        const oppositeSide = this.exerciseSlug === 'curl-left' ? 'right' : 'left';
        const tracks = clip.tracks.map((track) => (
            this.isArmTrackForSide(track.name, oppositeSide)
                ? this.freezeTrackAtFirstPose(track)
                : track.clone()
        ));
        return new THREE.AnimationClip(clip.name, clip.duration, tracks, clip.blendMode);
    }

    clearActiveAnimation() {
        if (this.mixer && this.mixerFinishedHandler) {
            this.mixer.removeEventListener('finished', this.mixerFinishedHandler);
        }
        this.mixerFinishedHandler = null;
        if (this.activeAction) this.activeAction.stop();
        if (this.mixer) this.mixer.stopAllAction();
        this.activeAction = null;
        this.mixer = null;
        if (this.avatar) this.scene.remove(this.avatar);
        this.avatar = null;
    }

    activateAsset(asset, animationKey, { loop = true, stage = 'main' } = {}) {
        if (!asset?.model || !asset?.clips?.length) return false;
        this.clearActiveAnimation();
        this.avatar = asset.model;
        this.scene.add(this.avatar);
        this.fitObject(this.avatar);

        const rawClip = this.selectClip(asset.clips, animationKey);
        const clip = this.prepareClipForExercise(rawClip);
        this.mixer = new THREE.AnimationMixer(this.avatar);
        this.activeAction = this.mixer.clipAction(clip);
        this.activeAction.reset();
        this.activeAction.enabled = true;
        this.activeAction.paused = !this.playing;
        this.mixer.timeScale = this.speed;
        this.activeStage = stage;

        if (loop) {
            this.activeAction.setLoop(THREE.LoopRepeat, Infinity);
            this.activeAction.clampWhenFinished = false;
        } else {
            this.activeAction.setLoop(THREE.LoopOnce, 1);
            this.activeAction.clampWhenFinished = true;
            this.mixerFinishedHandler = (event) => {
                if (event.action !== this.activeAction || this.activeStage !== 'intro') return;
                this.startMainLoop();
            };
            this.mixer.addEventListener('finished', this.mixerFinishedHandler);
        }
        this.activeAction.play();
        return { clip };
    }

    startSequence() {
        if (!this.mainAsset) return;
        this.playing = true;
        if (this.playButton) this.playButton.textContent = 'Tạm dừng';

        if (this.introAsset) {
            const result = this.activateAsset(this.introAsset, this.introAnimationKey, { loop: false, stage: 'intro' });
            if (result) {
                // Push-up intro and main clip are one exercise sequence. The
                // intro runs once; the finished event switches to main loop.
                this.setStatus('', 'neutral');
                return;
            }
        }
        this.startMainLoop();
    }

    startMainLoop() {
        if (!this.mainAsset) return;
        const result = this.activateAsset(this.mainAsset, this.animationKey, { loop: true, stage: 'main' });
        if (!result) {
            this.setControlsEnabled(false);
            this.setStatus('Mô hình chính chưa có animation clip hợp lệ.', 'error');
            return;
        }
        this.setStatus('', 'neutral');
    }

    fitObject(object) {
        // Reset transform offsets applied by a previous activation before
        // fitting the same cached scene again after the user presses Reset.
        if (!object.userData.fitMotionOriginalTransform) {
            object.userData.fitMotionOriginalTransform = {
                position: object.position.clone(),
                scale: object.scale.clone(),
            };
        } else {
            object.position.copy(object.userData.fitMotionOriginalTransform.position);
            object.scale.copy(object.userData.fitMotionOriginalTransform.scale);
        }

        const box = new THREE.Box3().setFromObject(object);
        const size = box.getSize(new THREE.Vector3());
        const height = Math.max(size.y, 0.01);
        const scale = 1.9 / height;
        object.scale.multiplyScalar(scale);

        const scaledBox = new THREE.Box3().setFromObject(object);
        const scaledCenter = scaledBox.getCenter(new THREE.Vector3());
        object.position.x -= scaledCenter.x;
        object.position.z -= scaledCenter.z;
        object.position.y -= scaledBox.min.y;

        this.controls.target.set(0, 1.0, 0);
        this.controls.update();
    }

    resize() {
        const width = Math.max(280, this.canvasHost.clientWidth || 640);
        const height = Math.max(300, this.canvasHost.clientHeight || 480);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height, false);
    }

    animate() {
        const delta = Math.min(this.clock.getDelta(), 0.05);
        // AnimationMixer already respects timeScale. Do not multiply by speed
        // again, otherwise 1.5x would accidentally run at 2.25x.
        if (this.playing && this.mixer) {
            this.mixer.update(delta);
        }
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
        requestAnimationFrame(this.animate);
    }
}

document.querySelectorAll('[data-exercise-model-viewer]').forEach((root) => {
    try {
        new ExerciseModelViewer(root);
    } catch (error) {
        console.error('Không khởi tạo được trình xem 3D', error);
        const status = root.querySelector('[data-viewer-status]');
        if (status) status.textContent = 'Trình duyệt không khởi tạo được WebGL.';
    }
});
