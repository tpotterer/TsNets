import { expect } from "chai";
import { Atom } from "../../shared/atom/atom.class";
import { ConvLayer } from "./conv-layer.class";

describe("conv layer", () => {
  it("should create a conv layer", () => {
    const convLayer = new ConvLayer(1, 3, 1);
    expect(convLayer).to.not.be.undefined;
  });
  it("should handle forward pass with no stride", () => {
    const convLayer = new ConvLayer(2, 3, 1);
    // 2 channel input
    const input = [
      [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16],
        [17, 18, 19, 20, 21, 22, 23, 24],
      ],
      [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16],
        [17, 18, 19, 20, 21, 22, 23, 24],
      ],
    ];
    const output: any = convLayer.forward(
      input.map((x) => x.map((elem) => elem.map((i) => new Atom(i)))),
      null as any
    );
    expect(output).to.not.be.undefined;
    expect(output.length).to.eql(3);
    expect(output[0].length).to.eql(3);
    expect(output[0][0].length).to.eql(8);
  });
  it("should handle forward pass with stride", () => {
    const convLayer = new ConvLayer(2, 3, 2);
    // 2 channel input
    const input = [
      [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16],
        [17, 18, 19, 20, 21, 22, 23, 24],
      ],
      [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16],
        [17, 18, 19, 20, 21, 22, 23, 24],
      ],
    ];
    const output: any = convLayer.forward(
      input.map((x) => x.map((elem) => elem.map((i) => new Atom(i)))),
      null as any
    );
    expect(output).to.not.be.undefined;
    expect(output.length).to.eql(3);
    expect(output[0].length).to.eql(2);
    expect(output[0][0].length).to.eql(4);
  });
  it("should handle forward pass 2", () => {
    const convLayer = new ConvLayer(1, 6, 1);
    // 2 channel input
    const input = [
      new Array(28)
        .fill(0)
        .map((elem) => new Array(28).fill(0).map((elem) => Math.random())),
    ];
    const output: any = convLayer.forward(
      input.map((x) => x.map((elem) => elem.map((i) => new Atom(i)))),
      null as any
    );
    expect(output).to.not.be.undefined;
    expect(output.length).to.eql(6);
    expect(output[0].length).to.eql(28);
    expect(output[0][0].length).to.eql(28);
  });
  it("should handle forward pass with kernel 5", () => {
    const convLayer = new ConvLayer(1, 6, 1, 5);
    // 2 channel input
    const input = [
      new Array(28)
        .fill(0)
        .map((elem) => new Array(28).fill(0).map((elem) => Math.random())),
    ];
    const output: any = convLayer.forward(
      input.map((x) => x.map((elem) => elem.map((i) => new Atom(i)))),
      null as any
    );
    expect(output).to.not.be.undefined;
    expect(output.length).to.eql(6);
    expect(output[0].length).to.eql(28);
    expect(output[0][0].length).to.eql(28);
  });
});
