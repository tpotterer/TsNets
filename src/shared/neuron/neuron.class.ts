import { Atom } from "../atom/atom.class";

export class Neuron {
  private w_in: Atom[] = [];
  private b: Atom | null;

  constructor(private fanIn: number, private hasBias: boolean = false) {
    if (fanIn < 1) {
      throw new Error(`fanIn must be greater than 0`);
    }
    this.w_in = new Array(fanIn).fill(0).map(() => new Atom(Math.random()));

    this.b = !!this.hasBias ? new Atom(Math.random()) : null;
  }

  public forward(inputs: Atom[]): Atom {
    if (inputs.length !== this.fanIn) {
      throw new Error(`Inputs length must be equal to fanIn`);
    }
    const result = inputs
      .map((input, i) => input.mul(this.w_in[i]))
      .reduce((a, b) => a.add(b), this.b || new Atom(0));

    return result;
  }

  public getParameters(): Atom[] {
    return [...this.w_in, ...(this.b ? [this.b] : [])];
  }

  public setParameters(params: Atom[]): Atom[] {
    this.w_in = params.slice(0, this.fanIn);
    params = params.slice(this.fanIn);
    if (this.b) {
      this.b = params[0];
      params = params.slice(1);
    }
    return params;
  }
}
