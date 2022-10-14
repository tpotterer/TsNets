import { LinearLayer } from "../layers/linear/linear-layer.class";
import { NonLinearityLayer } from "../layers/non-linearity/non-linearity-layer.class";
import { ConvLayer } from "../layers/convolution/conv-layer.class";
import { AvgPoolingLayer } from "../layers/pooling/avg-pooling-layer.class";
import { ThreeDNonLinearityLayer } from "../layers/3d-operations/3d-non-linearity-layer.class";
import { FlattenLayer } from "../layers/3d-operations/flatten-layer.class";

export type AnyLayerType =
  | LinearLayer
  | NonLinearityLayer
  | ConvLayer
  | AvgPoolingLayer
  | ThreeDNonLinearityLayer
  | FlattenLayer;
