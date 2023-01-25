import { Atom } from "./atom/atom.class";
import { ModelModes } from "./model-modes.enum";
import { AtomTensor } from "./atom-tensor.type";
import { LayerTypes } from "./layer-types.enum";

export abstract class LayerBase {
  public abstract layerType: LayerTypes;

  abstract forward(inputs: AtomTensor, modelMode: ModelModes): AtomTensor;

  abstract getParameters(): Atom[];

  abstract setParameters(params: Atom[]): Atom[];
}
