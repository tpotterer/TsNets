import { Atom } from "../../shared/atom/atom.class";
import { LayerBase } from "../../shared/layer-base.class";
import { ModelModes } from "../../shared/model-modes.enum";
import { AtomTensor } from "../../shared/atom-tensor.type";

export class FlattenLayer extends LayerBase {
  constructor() {
    super();
  }

  public forward(atomsIn: AtomTensor, modelMode: ModelModes): AtomTensor {
    const inputs: Atom[][][] = atomsIn as any;
    return inputs.flat(3);
  }

  public getParameters(): Atom[] {
    return [];
  }

  public setParameters(params: Atom[]): Atom[] {
    return params;
  }
}
