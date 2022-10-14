import { expect } from "chai";
import { Atom } from "../atom/atom.class";
import { Neuron } from "./neuron.class";
import { NonLinearityTypes } from "../non-linearity-types.enum";

describe("neuron tests", () => {
  it("should create a neuron", () => {
    const neuron = new Neuron(2);
    expect(neuron).to.instanceOf(Neuron);
  });

  it("should run forward without bias", () => {
    const neuron = new Neuron(2);
    const inputs = [0.1, 0.5];
    const result = neuron.forward(inputs.map((elem) => new Atom(elem)));
    expect(result).to.be.instanceOf(Atom);
    expect(neuron.getParameters().length).to.eql(2);
  });

  it("should run forward with bias", () => {
    const neuron = new Neuron(2, true);
    const inputs = [0.1, 0.5];
    const result = neuron.forward(inputs.map((elem) => new Atom(elem)));
    expect(result).to.be.instanceOf(Atom);
    expect(neuron.getParameters().length).to.eql(3);
  });
});
