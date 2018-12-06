let dados;
let model;
let xs, ys;
let rSlider, gSlider, bSlider;
let labelP;
let lossP;
let canvas;
let grafico;
let btnSalvar, btnCarregar, btnTreinar;
let perdaX = [];
let perdaY = [];
let accY = [];
let treinando;

let listaLabels = [
  "red-ish",
  "green-ish",
  "blue-ish",
  "orange-ish",
  "yellow-ish",
  "pink-ish",
  "purple-ish",
  "brown-ish",
  "grey-ish"
];

function parsePt(key) {
  const translator = {
    "red-ish": "Tom de vermelho",
    "green-ish": "Tom de verde",
    "blue-ish": "Tom de azul",
    "orange-ish": "Tom de laranja",
    "yellow-ish": "Tom de amarelo",
    "pink-ish": "Tom de rosa",
    "purple-ish": "Tom de roxo",
    "brown-ish": "Tom de marrom",
    "grey-ish": "Tom de cinza"
  };

  return translator[key] || "Não identificado";
}

function preload() {
  dados = loadJSON("./colorData.json");
}

async function loadMd() {
  if (localStorage.length > 0) {
    const LEARNING_RATE = 0.25;
    const optimizer = tf.train.sgd(LEARNING_RATE);
    let item = Number(localStorage.getItem("saveNo"));
    model.compile({
      optimizer: optimizer,
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"]
    });
    console.log("Modelo carregado");
  } else {
    alert("No previous models saved!");
  }
}

async function saveModel() {
  if (localStorage.length > 0) {
    let item = Number(localStorage.getItem("saveNo"));
    await model.save(`indexeddb://colorClassifier-${item + 1}`);
    localStorage.setItem("saveNo", item + 1);
  } else {
    await model.save(`indexeddb://colorClassifier-1`);
    localStorage.setItem("saveNo", 1);
  }
}

function setup() {
  canvas = createCanvas(200, 200);
  grafico = document.getElementById("graph");
  labelP = select("#prediction");
  lossP = select("#loss");
  rSlider = select("#red-slider");
  gSlider = select("#green-slider");
  bSlider = select("#blue-slider");
  btnSalvar = select("#save");
  btnCarregar = select("#load");
  btnTreinar = select("#train");

  canvas.parent("rgb-Canvas");
  let cores = [];
  let labels = [];
  for (let record of dados.entries) {
    let col = [record.r / 255, record.g / 255, record.b / 255];
    cores.push(col);
    labels.push(listaLabels.indexOf(record.label));
  }

  xs = tf.tensor2d(cores);
  let tensorNomes = tf.tensor1d(labels, "int32");

  ys = tf.oneHot(tensorNomes, 9).cast("float32");
  tensorNomes.dispose();

  model = buildModel();
  btnSalvar.mouseClicked(saveModel);
  btnCarregar.mouseClicked(loadMd);

  treinando = false;
  btnTreinar.mouseClicked(train);
}

async function train() {
  if (treinando) {
    return;
  }
  treinando = true;
  alert("Treinamento iniciado!");
  await model.fit(xs, ys, {
    shuffle: true,
    validationSplit: 0.05,
    epochs: 12,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log("Época finalizada");
        const trainingDiv = document.getElementById("training");
        trainingDiv.innerHTML += `<h5 style="margin-bottom: 4px;">Época Finalizada!</h5>`;
        perdaY.push(logs.val_loss.toFixed(2));
        accY.push(logs.val_acc.toFixed(2));
        perdaX.push(perdaX.length + 1);
        lossP.html("Perda: " + logs.loss.toFixed(5));
      },
      onBatchEnd: async (batch, logs) => {
        console.log("bloco finalizado");
        await tf.nextFrame();
      },
      onTrainEnd: () => {
        treinando = false;
        const trainingDiv  = document.getElementById("training");
        trainingDiv.innerHTML += `<h3 style="color: green">Treinamento Finalizado!</h3>`;
        console.log("Treinamento finalizado");
      }
    }
  });
}

function buildModel() {
  let md = tf.sequential();
  const hidden = tf.layers.dense({
    units: 15,
    inputShape: [3],
    activation: "sigmoid"
  });

  const output = tf.layers.dense({
    units: 9,
    activation: "softmax"
  });

  md.add(hidden);
  md.add(output);

  const TAXA_APRENDIZADO = 0.25;
  const optimizer = tf.train.sgd(TAXA_APRENDIZADO);

  md.compile({
    optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  return md;
}

function plotTraining() {
  let layout = {
    width: 600,
    height: 300,
    title: 'Gráfico de progresso',
    xaxis: {
      title: 'N. de Épocas'
    }
  };

  let prec = {
    x: perdaX,
    y: accY,
    name: "Precisão"
  };

  let perda = {
    x: perdaX,
    y: perdaY,
    name: 'Perda'
  };

  // Plotly.plot(graph, [perda, prec], layout);
}

function draw() {
  let r = rSlider.value();
  let g = gSlider.value();
  let b = bSlider.value();
  background(r, g, b);

  tf.tidy(() => {
    const input = tf.tensor2d([[r, g, b]]);
    let results = model.predict(input);
    let argMax = results.argMax(1);
    let index = argMax.dataSync()[0];
    let label = listaLabels[index];
    const final = parsePt(label);
    labelP.html("Cor: " + final);
  });

  plotTraining();
}
