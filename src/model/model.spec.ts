import { expect } from "chai";
import { Atom } from "../shared/atom/atom.class";
import { LinearLayer } from "../layers/linear/linear-layer.class";
import { NonLinearityLayer } from "../layers/non-linearity/non-linearity-layer.class";
import { ModelModes } from "../shared/model-modes.enum";
import { NonLinearityTypes } from "../shared/non-linearity-types.enum";
import { Model } from "./model.class";
import { ConvLayer } from "../layers/convolution/conv-layer.class";
import { AvgPoolingLayer } from "../layers/pooling/avg-pooling-layer.class";
import { ThreeDNonLinearityLayer } from "../layers/3d-operations/3d-non-linearity-layer.class";
import { FlattenLayer } from "../layers/3d-operations/flatten-layer.class";

describe("Model", () => {
  it("simple linear model", () => {
    const model = new Model([
      new LinearLayer(2, 3),
      new NonLinearityLayer(NonLinearityTypes.Tanh),
      new LinearLayer(3, 1),
    ]);

    const input = [1, 2].map((e) => new Atom(e));

    const output = model.forward(input);
    expect(output.length).to.eql(1);
  });

  it("simple linear model with dropout", () => {
    // all nodes dropout
    const model = new Model([
      new LinearLayer(2, 3, false, 1),
      new NonLinearityLayer(NonLinearityTypes.Tanh),
      new LinearLayer(3, 1, false, 1),
    ]);

    const input = [1, 2].map((e) => new Atom(e));

    // dropout present in training mode
    const trainOutput = model.forward(input) as Atom[];
    expect(trainOutput[0].value).to.eql(0);

    model.setMode(ModelModes.Inference);

    // dropout not present in inference mode
    const evalOutput = model.forward(input);
    expect(evalOutput[0]).to.not.eql(0);
  });

  it("training test", () => {
    const xs = [
      [new Atom(2.0), new Atom(3.0), new Atom(-1.0)],
      [new Atom(3.0), new Atom(-1.0), new Atom(0.5)],
      [new Atom(0.5), new Atom(1.0), new Atom(1.0)],
      [new Atom(1.0), new Atom(1.0), new Atom(-1.0)],
    ];
    const ys = [new Atom(1.0), new Atom(-1.0), new Atom(-1.0), new Atom(1.0)];

    const model = new Model(
      [
        new LinearLayer(3, 4),
        new NonLinearityLayer(NonLinearityTypes.Tanh),
        new LinearLayer(4, 4),
        new NonLinearityLayer(NonLinearityTypes.ReLU),
        new LinearLayer(4, 1),
      ],
      ModelModes.Training
    );

    let first_loss;
    let last_loss;
    for (let i = 0; i < 300; i++) {
      const ypred = xs.map((elem) => model.forward(elem) as Atom[]);
      const loss = ypred
        .map((elem, i) => elem[0].sub(ys[i]).pow(2))
        .reduce((a, b) => a.add(b));

      last_loss = loss.value;
      if (i === 0) {
        first_loss = loss.value;
      }

      model.getParameters().forEach((param) => (param.grad = 0.0));
      loss.backward();

      model
        .getParameters()
        .forEach((param) => (param.value += -0.1 * param.grad));
    }

    // this is a stupid test, but just want to show some training has been done
    expect(!!first_loss).to.be.true;
    expect(!!last_loss).to.be.true;
    if (first_loss && last_loss) {
      expect(first_loss * 0.5 > last_loss).eql(true);
    }
  });

  it("training test 2", () => {
    const xs = [
      [new Atom(2.0), new Atom(3.0), new Atom(-1.0)],
      [new Atom(3.0), new Atom(-1.0), new Atom(0.5)],
      [new Atom(0.5), new Atom(1.0), new Atom(1.0)],
      [new Atom(1.0), new Atom(1.0), new Atom(-1.0)],
    ];
    const ys = [new Atom(1.0), new Atom(-1.0), new Atom(-1.0), new Atom(1.0)];

    const model = new Model(
      [
        new LinearLayer(3, 4),
        new NonLinearityLayer(NonLinearityTypes.Tanh),
        new LinearLayer(4, 4, false, 0, true),
        new NonLinearityLayer(NonLinearityTypes.ReLU),
        new LinearLayer(4, 1),
      ],
      ModelModes.Training
    );

    let first_loss;
    let last_loss;
    for (let i = 0; i < 300; i++) {
      const ypred = xs.map((elem) => model.forward(elem) as Atom[]);

      const loss = ypred
        .map((elem, i) => elem[0].sub(ys[i]).pow(2))
        .reduce((a, b) => a.add(b));

      last_loss = loss.value;
      if (i === 0) {
        first_loss = loss.value;
      }

      model.getParameters().forEach((param) => (param.grad = 0.0));
      loss.backward();

      model
        .getParameters()
        .forEach((param) => (param.value += -0.1 * param.grad));
    }

    // this is a stupid test, but just want to show some training has been done
    expect(!!first_loss).to.be.true;
    expect(!!last_loss).to.be.true;
    if (first_loss && last_loss) {
      expect(first_loss * 0.5 > last_loss).eql(true);
    }
  });

  it("training test 3", () => {
    const x = [new Atom(2.0), new Atom(3.0), new Atom(-1.0)];
    const y = new Atom(1.0);

    const model = new Model(
      [
        new LinearLayer(3, 3, false, 0, true),
        new NonLinearityLayer(NonLinearityTypes.Tanh),
        new LinearLayer(3, 1, false, 0),
      ],
      ModelModes.Inference
    );

    let pred = model.forward(x) as Atom[];
    let loss = pred[0].sub(y).pow(2);

    model.getParameters().forEach((param) => (param.grad = 0.0));
    loss.backward();

    model
      .getParameters()
      .forEach((param) => (param.value += -0.1 * param.grad));

    pred = model.forward(x) as Atom[];
    expect(loss.value).to.be.greaterThan(pred[0].sub(y).pow(2).value);
  });

  it("convolution test", () => {
    // LeNet classifier
    const model = new Model([
      new ConvLayer(1, 6, 1, 5),
      new ThreeDNonLinearityLayer(NonLinearityTypes.ReLU),
      new AvgPoolingLayer(2, 2),
      new ConvLayer(6, 16, 1, 5),
      new ThreeDNonLinearityLayer(NonLinearityTypes.ReLU),
      new AvgPoolingLayer(2, 2),
      new FlattenLayer(),
      new LinearLayer(16 * 7 * 7, 120),
      new NonLinearityLayer(NonLinearityTypes.ReLU),
      new LinearLayer(120, 84),
      new NonLinearityLayer(NonLinearityTypes.ReLU),
      new LinearLayer(84, 10),
    ]);

    const input = [
      new Array(28)
        .fill(0)
        .map((elem) =>
          new Array(28).fill(0).map((elem) => new Atom(Math.random()))
        ),
    ];

    const output = model.forward(input) as Atom[];

    const expectedOutput = new Array(10).fill(0).map((elem) => new Atom(0.0));
    const totalLoss = output.reduce((prev, curr, idx) => {
      const loss = curr.sub(expectedOutput[idx]).pow(2);
      return prev.add(loss);
    }, new Atom(0.0));

    expect(totalLoss.value).to.be.greaterThan(0.0);
    expect(output.length).to.be.equal(10);
  });

  it("linear layer weight loading", () => {
    const model = new Model([
      new LinearLayer(3, 4),
      new NonLinearityLayer(NonLinearityTypes.Tanh),
      new LinearLayer(4, 4, false, 0, true),
      new NonLinearityLayer(NonLinearityTypes.ReLU),
      new LinearLayer(4, 1),
    ]);

    const parameters = model.getParameters();

    const rawWeights = parameters.map((param) => param.value);

    // set all parameters to 0
    const newWeights = new Array(rawWeights.length).fill(0);
    model.setParameters(newWeights);

    // check that worked
    expect(model.getParameters().map((param) => param.value)).to.eql(
      newWeights
    );

    // set them back to what they were before
    model.setParameters(rawWeights);
    expect(model.getParameters().map((param) => param.value)).to.eql(
      rawWeights
    );
  });

  it("conv layer weight loading", () => {
    // not bothered about the model making sense, just want to test the loading
    const model = new Model([
      new ConvLayer(1, 10, 2, 5),
      new ConvLayer(1, 4, 2, 5),
      new ConvLayer(1, 2, 2, 5),
      new LinearLayer(40, 10, true),
    ]);

    const parameters = model.getParameters();

    const rawWeights = parameters.map((param) => param.value);

    // set all parameters to 0
    const newWeights = new Array(rawWeights.length).fill(0);
    model.setParameters(newWeights);

    // check that worked
    expect(model.getParameters().map((param) => param.value)).to.eql(
      newWeights
    );

    // set them back to what they were before
    model.setParameters(rawWeights);

    expect(model.getParameters().map((param) => param.value)).to.eql(
      rawWeights
    );
  });
});
