declare module 'n8ao' {
  import type { Camera, Color, Scene } from 'three';
  import { Pass } from 'postprocessing';
  export class N8AOPostPass extends Pass {
    constructor(scene: Scene, camera: Camera, width?: number, height?: number);
    configuration: {
      aoSamples: number;
      aoRadius: number;
      denoiseSamples: number;
      denoiseRadius: number;
      distanceFalloff: number;
      intensity: number;
      denoiseIterations: number;
      renderMode: number;
      biasOffset: number;
      biasMultiplier: number;
      color: Color;
      gammaCorrection: boolean;
      screenSpaceRadius: boolean;
      halfRes: boolean;
      depthAwareUpsampling: boolean;
      colorMultiply: boolean;
      transparencyAware: boolean;
      accumulate: boolean;
    };
    setQualityMode(mode: 'Performance' | 'Low' | 'Medium' | 'High' | 'Ultra'): void;
    setDisplayMode(mode: string): void;
  }
}
