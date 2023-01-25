import { Neuron } from "../../shared/neuron/neuron.class";
import { Atom } from "../../shared/atom/atom.class";
import { ModelModes } from "../../shared/model-modes.enum";
import { LayerBase } from "../../shared/layer-base.class";
import { AtomTensor } from "../../shared/atom-tensor.type";
import { LayerTypes } from "../../shared/layer-types.enum";

export class LinearLayer extends LayerBase {
  public layerType: LayerTypes = LayerTypes.Linear;
  private neurons: Neuron[] = [];
  private dropoutMask: boolean[] = [];

  constructor(
    private layerIn: number,
    private layerOut: number,
    private hasBias: boolean = false,
    private dropoutRate: number = 0,
    private useResidual: boolean = false
  ) {
    super();
    if (this.layerIn < 1 || this.layerOut < 1) {
      throw new Error(`layerIn and layerOut must be greater than 0`);
    }
    if (useResidual && layerIn !== layerOut) {
      throw new Error(
        `layerIn and layerOut must be equal when using residual connections`
      );
    }
    this.dropoutRate = Math.min(Math.max(this.dropoutRate, 0), 1);
    this.neurons = new Array(this.layerOut)
      .fill(0)
      .map(() => new Neuron(this.layerIn, this.hasBias));
  }

  public forward(atomsIn: AtomTensor, modelMode: ModelModes): AtomTensor {
    const inputs: Atom[] = atomsIn as any;
    if (inputs.length !== this.layerIn) {
      throw new Error(`Inputs length must be equal to layerIn`);
    }
    // true = drop, false = keep
    this.dropoutMask = this.neurons.map(() => Math.random() < this.dropoutRate);
    const output = this.neurons.map((neuron, idx) =>
      modelMode !== ModelModes.Training || !this.dropoutMask[idx]
        ? neuron.forward(inputs)
        : new Atom(0)
    );
    return output.map((elem, idx) =>
      this.useResidual ? elem.add(inputs[idx]) : elem
    );
  }

  public setParameters(params: Atom[]): Atom[] {
    this.neurons.forEach((neuron: Neuron) => {
      params = neuron.setParameters(params);
    });
    return params;
  }

  public getParameters(): Atom[] {
    return this.neurons.reduce(
      (a: Atom[], b: Neuron) => [...a, ...b.getParameters()],
      []
    );
  }

  public getLastDropoutMask(): boolean[] {
    return this.dropoutMask;
  }

  public getNeurons(): Neuron[] {
    return [...this.neurons];
  }
}
