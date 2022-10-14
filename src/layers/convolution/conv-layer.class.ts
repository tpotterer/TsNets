import { Atom } from "../../shared/atom/atom.class";
import { LayerBase } from "../../shared/layer-base.class";
import { ModelModes } from "../../shared/model-modes.enum";
import { AtomTensor } from "../../shared/atom-tensor.type";

export class ConvLayer extends LayerBase {
  // (4d array)
  // (output channels, (3x3 kernel, (input channels, (kxk matrix))))
  private kernels: Atom[][][][] = [];

  constructor(
    private inputChannels: number,
    private outputChannels: number,
    private stride: number,
    private kernelSize: number = 3
  ) {
    super();
    if (kernelSize % 2 === 0) {
      throw new Error("Kernel size must be odd");
    }
    this.kernels = new Array(this.outputChannels).fill(0).map(() => {
      return new Array(this.inputChannels).fill(0).map(() => {
        return new Array(this.kernelSize)
          .fill(0)
          .map(() =>
            new Array(this.kernelSize)
              .fill(0)
              .map(() => new Atom(Math.random()))
          );
      });
    });
  }

  public forward(atomsIn: AtomTensor, modelMode: ModelModes): AtomTensor {
    const inputs: Atom[][][] = atomsIn as any;
    // input expected in the form of (input channels, (y, x))
    if (inputs.length !== this.inputChannels) {
      throw new Error(
        `Expected ${this.inputChannels} input channels, got ${inputs.length}`
      );
    }
    const output = this.kernels
      .map((kernel) => {
        return inputs.map((singleChannelImage, inputChannel) => {
          const channelKernel = kernel[inputChannel];

          const padded_inputs = [...singleChannelImage.map((row) => [...row])];

          const result = [];
          for (let y = 0; y < padded_inputs.length; y += this.stride) {
            const row: Atom[] = [];
            for (let x = 0; x < padded_inputs[0].length; x += this.stride) {
              let sum = new Atom(0);
              for (let ky = 0; ky < channelKernel.length; ky++) {
                for (let kx = 0; kx < channelKernel[ky].length; kx++) {
                  const v = padded_inputs[y + ky - 1];
                  const a = !!v ? v[x + kx - 1] || new Atom(0) : new Atom(0);
                  sum = sum.add(channelKernel[ky][kx].mul(a));
                }
              }
              row.push(sum);
            }
            result.push(row);
          }
          return result;
        });
      })
      // perform convolution for each input channel
      // convolving three kernel over a one channel image
      // (no. of kernels used, input no, number of rows, number of columns)
      // must sum over input dimensions
      .map((outChannel) => {
        return outChannel.reduce((acc, curr, idx) => {
          if (idx === 0) {
            return curr;
          }
          return acc.map((row, y) => {
            return row.map((val, x) => {
              return val.add(curr[y][x]);
            });
          });
        }, outChannel[0]);
      });

    // (output channels, rows, columns)
    return output;
  }

  public getParameters(): Atom[] {
    return [...this.kernels.flat(3)];
  }

  public setParameters(params: Atom[]): Atom[] {
    // take enough parameters for all kernels
    const layerParams = params.slice(0, this.kernels.flat(3).length);

    this.kernels.forEach((kernel) => {
      // take enough parameters for one kernel
      const kernelParams = layerParams.splice(0, kernel.flat(2).length);
      kernel.forEach((channel) => {
        // take enough params for this channel
        const channelParams = kernelParams.splice(0, channel.flat(1).length);
        channel.forEach((row) => {
          // take enough params for this row
          const rowParams = channelParams.splice(0, row.length);
          row.forEach((atom, idx) => {
            atom.value = rowParams[idx].value;
          });
        });
      });
    });

    return params.slice(this.kernels.flat(3).length);
  }
}
