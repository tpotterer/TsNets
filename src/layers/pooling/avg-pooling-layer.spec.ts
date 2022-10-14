import { expect } from "chai";
import { Atom } from "../../shared/atom/atom.class";
import { ConvLayer } from "../convolution/conv-layer.class";
import { AvgPoolingLayer } from "./avg-pooling-layer.class";

describe("avg pooling", () => {
  it("simple test", () => {
    const convLayer = new ConvLayer(1, 6, 1, 5);
    // 2 channel input
    const input = [
      new Array(28)
        .fill(0)
        .map((elem) => new Array(28).fill(0).map((elem) => Math.random())),
    ];
    const output = convLayer.forward(
      input.map((x) => x.map((elem) => elem.map((i) => new Atom(i)))),
      null as any
    );

    const avgPoolLayer = new AvgPoolingLayer(2, 2);

    const pooledOutput = avgPoolLayer.forward(
      output,
      null as any
    ) as Atom[][][];

    expect(pooledOutput).to.not.be.undefined;
    expect(pooledOutput.length).to.eql(6);
    expect(pooledOutput[0].length).to.eql(14);
    expect(pooledOutput[0][0].length).to.eql(14);
  });
});
