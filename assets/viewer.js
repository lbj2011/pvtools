// Load Three.js automatically from CDN
const script1 = document.createElement('script');
script1.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
document.head.appendChild(script1);

const script2 = document.createElement('script');
script2.src = "https://cdn.jsdelivr.net/npm/three@0.128/examples/js/loaders/GLTFLoader.js";
document.head.appendChild(script2);

script2.onload = () => {

    const container = document.getElementById("viewer");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    const camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    // Light
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(100, 100, 100);
    scene.add(light);

    const ambient = new THREE.AmbientLight(0xaaaaaa);
    scene.add(ambient);

    // Load GLB
    const loader = new THREE.GLTFLoader();
    loader.load('/assets/Untitled.glb', function (gltf) {
        scene.add(gltf.scene);
    });

    camera.position.z = 200;

    function animate() {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    }
    animate();
};