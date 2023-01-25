import { Atom } from "../shared/atom/atom.class";
import { AnyLayerType } from "../shared/any-layer-type.type";
import { ModelModes } from "../shared/model-modes.enum";
import { AtomTensor } from "../shared/atom-tensor.type";

export class Model {
  constructor(
    private layers: AnyLayerType[],
    private modelMode: ModelModes = ModelModes.Training
  ) {}

  public forward(inputs: AtomTensor): AtomTensor {
    return this.layers.reduce((runningResult, layer) => {
      return layer.forward(runningResult, this.modelMode);
    }, inputs);
  }

  public setMode(mode: ModelModes): void {
    this.modelMode = mode;
  }

  public getParameters(): Atom[] {
    return this.layers.reduce(
      (acc: Atom[], layer) => [...acc, ...layer.getParameters()],
      []
    );
  }

  public setParameters(rawParams: number[]): void {
    let params: Atom[] = rawParams.map((elem) => new Atom(elem));
    this.layers.forEach((layer) => {
      params = layer.setParameters(params);
    });
  }

  public getLayers(): AnyLayerType[] {
    return this.layers;
  }
}
