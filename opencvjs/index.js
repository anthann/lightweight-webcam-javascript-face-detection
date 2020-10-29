let videoWidth, videoHeight;

// whether streaming video from the camera.
let streaming = false;

let video = document.getElementById('video');
let canvasOutput = document.getElementById('canvasOutput');
let canvasOutputCtx = canvasOutput.getContext('2d');
let stream = null;

let detectFace = document.getElementById('face');
let detectEye = document.getElementById('eye');
let playButton = document.getElementById("playbutton");

let faceClassifier = null;
let eyeClassifier = null;

let videoCapture;
let srcMat;
let grayMat;
let _cv = undefined;

playButton.addEventListener("click", function() {
  if (!(typeof _cv === 'undefined')) {
    initUI();
    startCamera();
  }
}, false);


function startCamera() {
  if (streaming) return;
  navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then(function (s) {
      stream = s;
      video.srcObject = s;
      video.play();
    })
    .catch(function (err) {
      console.log("An error occured! " + err);
    });

  video.addEventListener("canplay", function (ev) {
    if (!streaming) {
      videoWidth = video.videoWidth;
      videoHeight = video.videoHeight;
      video.setAttribute("width", videoWidth);
      video.setAttribute("height", videoHeight);
      video.setAttribute('autoplay', '');
      video.setAttribute('muted', '');
      video.setAttribute('playsinline', '');
      canvasOutput.width = videoWidth;
      canvasOutput.height = videoHeight;
      streaming = true;
    }
    startVideoProcessing();
  }, false);
}

function startVideoProcessing() {
  if (!streaming) { console.warn("Please startup your webcam"); return; }
  videoCapture = new _cv.VideoCapture(video)

  srcMat = new _cv.Mat(videoHeight, videoWidth, _cv.CV_8UC4);
  grayMat = new _cv.Mat(videoHeight, videoWidth, _cv.CV_8UC1);

  faceClassifier = new _cv.CascadeClassifier();
  faceClassifier.load('haarcascade_frontalface_default.xml');

  eyeClassifier = new _cv.CascadeClassifier();
  eyeClassifier.load('haarcascade_eye.xml');

  requestAnimationFrame(processVideo);
}

function processVideo() {
  stats.begin();
  videoCapture.read(srcMat)
  _cv.cvtColor(srcMat, grayMat, _cv.COLOR_RGBA2GRAY);
  let faces = [];
  let eyes = [];
  let size;
  if (detectFace.checked) {
    let faceVect = new _cv.RectVector();
    let faceMat = new _cv.Mat();
    if (detectEye.checked) {
      _cv.pyrDown(grayMat, faceMat);
      size = faceMat.size();
    } else {
      _cv.pyrDown(grayMat, faceMat);
      _cv.pyrDown(faceMat, faceMat);
      size = faceMat.size();
    }
    faceClassifier.detectMultiScale(faceMat, faceVect);
    for (let i = 0; i < faceVect.size(); i++) {
      let face = faceVect.get(i);
      faces.push(new _cv.Rect(face.x, face.y, face.width, face.height));
      if (detectEye.checked) {
        let eyeVect = new _cv.RectVector();
        let eyeMat = faceMat.roi(face);
        eyeClassifier.detectMultiScale(eyeMat, eyeVect);
        for (let i = 0; i < eyeVect.size(); i++) {
          let eye = eyeVect.get(i);
          eyes.push(new _cv.Rect(face.x + eye.x, face.y + eye.y, eye.width, eye.height));
        }
        eyeMat.delete();
        eyeVect.delete();
      }
    }
    faceMat.delete();
    faceVect.delete();
  } else {
    if (detectEye.checked) {
      let eyeVect = new _cv.RectVector();
      let eyeMat = new _cv.Mat();
      _cv.pyrDown(grayMat, eyeMat);
      size = eyeMat.size();
      eyeClassifier.detectMultiScale(eyeMat, eyeVect);
      for (let i = 0; i < eyeVect.size(); i++) {
        let eye = eyeVect.get(i);
        eyes.push(new _cv.Rect(eye.x, eye.y, eye.width, eye.height));
      }
      eyeMat.delete();
      eyeVect.delete();
    }
  }
  _cv.imshow(canvasOutput, srcMat)
  drawResults(canvasOutputCtx, faces, 'red', size);
  drawResults(canvasOutputCtx, eyes, 'yellow', size);
  stats.end();
  requestAnimationFrame(processVideo);
}

function drawResults(ctx, results, color, size) {
  for (let i = 0; i < results.length; ++i) {
    let rect = results[i];
    let xRatio = videoWidth / size.width;
    let yRatio = videoHeight / size.height;
    ctx.lineWidth = 3;
    ctx.strokeStyle = color;
    ctx.strokeRect(rect.x * xRatio, rect.y * yRatio, rect.width * xRatio, rect.height * yRatio);
  }
}

function stopCamera() {
  if (!streaming) return;
  document.getElementById("canvasOutput").getContext("2d").clearRect(0, 0, width, height);
  video.pause();
  video.srcObject = null;
  stream.getVideoTracks()[0].stop();
  streaming = false;
}

function initUI() {
  stats = new Stats();
  stats.showPanel(0);
  document.getElementById('container').appendChild(stats.dom);
}

function opencvIsReady() {
  console.log('OpenCV.js is ready');
  cv.then(res => {
    _cv = res
    playButton.disabled = false
  })

}