import { Atom } from "./atom/atom.class";
import { ModelModes } from "./model-modes.enum";
import { AtomTensor } from "./atom-tensor.type";

export abstract class LayerBase {
  abstract forward(inputs: AtomTensor, modelMode: ModelModes): AtomTensor;

  abstract getParameters(): Atom[];

  abstract setParameters(params: Atom[]): Atom[];
}
