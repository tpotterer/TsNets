import { expect } from "chai";
import { Atom } from "./atom.class";

describe("Atom basic tests", () => {
  it("should create an instance", () => {
    expect(new Atom(0, "a")).to.be.instanceOf(Atom);
  });

  it("should add two atoms", () => {
    const a = new Atom(1, "a");
    const b = new Atom(2, "b");
    const c = a.add(b);
    expect(c.value).to.equal(3);
  });

  it("should sub two atoms", () => {
    const a = new Atom(1, "a");
    const b = new Atom(2, "b");
    const c = a.sub(b);
    expect(c.value).to.equal(-1);
  });

  it("should mul two atoms", () => {
    const a = new Atom(1, "a");
    const b = new Atom(2, "b");
    const c = a.mul(b);
    expect(c.value).to.equal(2);
  });

  it("should div two atoms", () => {
    const a = new Atom(1, "a");
    const b = new Atom(2, "b");
    const c = a.div(b);
    expect(c.value).to.equal(0.5);
  });

  it("should pow two atoms", () => {
    const a = new Atom(2, "a");
    const c = a.pow(3);
    expect(c.value).to.equal(8);
  });
});

describe("backward tests", () => {
  it("simple backward", () => {
    const a = new Atom(0.1);
    const b = new Atom(0.5);
    const c = new Atom(-0.5);
    const d = a.mul(b);
    const e = d.add(c);
    const L = e.tanh();

    L.backward();

    expect(a.grad).eql(0.4110006146845269);
    expect(b.grad).eql(0.08220012293690539);
    expect(c.grad).eql(0.8220012293690538);
    expect(d.grad).eql(0.8220012293690538);
    expect(e.grad).eql(0.8220012293690538);
    expect(L.grad).eql(1);
  });

  it("value used twice example", () => {
    const a = new Atom(-2);
    const b = a.add(a);

    b.backward();

    expect(a.grad).eql(2);
  });

  it("example with sub and pow", () => {
    const a = new Atom(0.4);
    const b = new Atom(0.3);
    const c = 2;
    const d = b.sub(a);
    const e = d.pow(c);
    const f = e.tanh();

    f.backward();

    expect(a.grad).eql(0.19998000133325783);
    expect(b.grad).eql(-0.19998000133325783);
    expect(d.grad).eql(-0.19998000133325783);
    expect(e.grad).eql(0.9999000066662889);
    expect(f.grad).eql(1);
  });
});
