import { Atom } from "../../shared/atom/atom.class";
import { NonLinearityTypes } from "../../shared/non-linearity-types.enum";
import { LayerBase } from "../../shared/layer-base.class";
import { ModelModes } from "../../shared/model-modes.enum";
import { AtomTensor } from "../../shared/atom-tensor.type";
import { LayerTypes } from "../../shared/layer-types.enum";

export class ThreeDNonLinearityLayer extends LayerBase {
  public layerType: LayerTypes = LayerTypes.ThreeDNonLinearity;
  constructor(
    private nonLinearity: NonLinearityTypes = NonLinearityTypes.Tanh
  ) {
    super();
  }

  // this could certainly be done better...
  public forward(atomsIn: AtomTensor, modelMode: ModelModes): AtomTensor {
    const inputs: Atom[][][] = atomsIn as any;
    return inputs.map((input) => {
      return input.map((row) => {
        return row.map((atom) => {
          switch (this.nonLinearity) {
            case NonLinearityTypes.Tanh:
              return atom.tanh();
            // case NonLinearityTypes.Sigmoid:
            //   return input.sigmoid();
            case NonLinearityTypes.ReLU:
              return atom.relu();
            // case NonLinearityTypes.LeakyReLU:
            //   return input.leakyRelu();
            // case NonLinearityTypes.Softmax:
            //   return input.softmax();
            default:
              throw new Error(`Non linearity not supported`);
          }
        });
      });
    });
  }

  public getParameters(): Atom[] {
    return [];
  }

  public setParameters(params: Atom[]): Atom[] {
    return params;
  }
}
