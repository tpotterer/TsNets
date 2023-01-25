import { Atom } from "../../shared/atom/atom.class";
import { LayerBase } from "../../shared/layer-base.class";
import { ModelModes } from "../../shared/model-modes.enum";
import { AtomTensor } from "../../shared/atom-tensor.type";
import { LayerTypes } from "../../shared/layer-types.enum";

export class AvgPoolingLayer extends LayerBase {
  public layerType: LayerTypes = LayerTypes.AvgPooling;
  constructor(private kernelSize: number, private stride: number) {
    super();
  }

  public forward(atomsIn: AtomTensor, modelMode: ModelModes): AtomTensor {
    const inputs: Atom[][][] = atomsIn as any;
    return inputs.map((channel) => {
      const result = [];
      for (let i = 0; i < channel.length; i += this.stride) {
        const row: Atom[] = [];
        for (let j = 0; j < channel[0].length; j += this.stride) {
          let sum = new Atom(0);
          for (let k = 0; k < this.kernelSize; k++) {
            for (let l = 0; l < this.kernelSize; l++) {
              sum = sum.add(channel[i + k][j + l]);
            }
          }
          row.push(sum.div(new Atom(this.kernelSize * this.kernelSize)));
        }
        result.push(row);
      }
      return result;
    });
  }

  public getParameters(): Atom[] {
    return [];
  }

  public setParameters(params: Atom[]): Atom[] {
    return params;
  }
}
