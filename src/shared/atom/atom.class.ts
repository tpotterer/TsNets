import { OperationTypes } from "../operation-types.enum";

export class Atom {
  public grad: number = 0;
  private _backward: () => void = () => null;
  constructor(
    public value: number,
    public label: string = "",
    private _prev: Atom[] = [],
    private _operation: OperationTypes = OperationTypes.None
  ) {}

  public backward(): void {
    // topographical sort
    const topo: Atom[] = [];
    const visited = new Set<Atom>();
    const build_topo = (v: Atom) => {
      if (!visited.has(v)) {
        visited.add(v);
        for (const prev of v._prev) {
          build_topo(prev);
        }
        topo.push(v);
      }
    };

    build_topo(this);
    topo.reverse();
    this.grad = 1.0;
    for (const v of topo) {
      v._backward();
    }
  }

  public toString(): string {
    return `(Atom: ${this.value})`;
  }

  public add(other: Atom): Atom {
    const out = new Atom(
      this.value + other.value,
      ``,
      [this, other],
      OperationTypes.Add
    );

    // this + other = out

    // d(out)/d(this) = 1
    // d(L)/d(this) = d(out)/d(this) * d(L)/d(out)

    // d(out)/d(other) = 1
    // d(L)/d(other) = d(out)/d(other) * d(L)/d(out)
    const _backward = () => {
      this.grad += 1.0 * out.grad;
      other.grad += 1.0 * out.grad;
    };
    out._backward = _backward;

    return out;
  }

  public sub(other: Atom): Atom {
    const out = new Atom(
      this.value - other.value,
      ``,
      [this, other],
      OperationTypes.Sub
    );

    // this - other = out

    // d(out)/d(this) = 1
    // d(L)/d(this) = d(out)/d(this) * d(L)/d(out)

    // d(out)/d(other) = -1
    // d(L)/d(other) = d(out)/d(other) * d(L)/d(out)

    const _backward = () => {
      this.grad += 1.0 * out.grad;
      other.grad += -1.0 * out.grad;
    };
    out._backward = _backward;

    return out;
  }

  public mul(other: Atom): Atom {
    const out = new Atom(
      this.value * other.value,
      ``,
      [this, other],
      OperationTypes.Mul
    );

    // this * other = out

    // d(out)/d(this) = other
    // d(L)/d(this) = d(out)/d(this) * d(L)/d(out)

    // d(out)/d(other) = this
    // d(L)/d(other) = d(out)/d(other) * d(L)/d(out)

    const _backward = () => {
      this.grad += other.value * out.grad;
      other.grad += this.value * out.grad;
    };
    out._backward = _backward;

    return out;
  }

  public div(other: Atom): Atom {
    // no grad needed, pow and mul will handle
    return this.mul(other.pow(-1));
  }

  public pow(other: number): Atom {
    const out = new Atom(this.value ** other, "", [this], OperationTypes.Pow);

    // in this case other is a number so only need to calculate grad for this
    // this ** other = out

    // d(out)/d(this) = (other) * (this ** (other-1))
    // d(L)/d(this) = d(out)/d(this) * d(L)/d(out)

    const _backward = () => {
      this.grad += other * this.value ** (other - 1) * out.grad;
    };
    out._backward = _backward;

    return out;
  }

  // ACTIVATION FUNCTIONS

  public tanh(): Atom {
    const t = Math.tanh(this.value);
    const out = new Atom(t, "", [this], OperationTypes.Tanh);

    // tanh(this) = out

    // d(out)/d(this) = 1 - (t ** 2)
    // d(L)/d(this) = d(out)/d(this) * d(L)/d(out)

    const _backward = () => {
      this.grad += (1 - t ** 2) * out.grad;
    };
    out._backward = _backward;

    return out;
  }

  public relu(): Atom {
    const r = Math.max(0, this.value);
    const out = new Atom(r, "", [this], OperationTypes.Relu);

    // relu(this) = out

    // d(out)/d(this) = 1 if this > 0 else 0
    // d(L)/d(this) = d(out)/d(this) * d(L)/d(out)

    const _backward = () => {
      this.grad += (this.value > 0 ? 1 : 0) * out.grad;
    };
    out._backward = _backward;

    return out;
  }
}
