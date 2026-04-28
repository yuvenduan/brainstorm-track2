'use client';

import { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

interface ChannelCoord {
  x: number;
  y: number;
}

interface Brain3DProps {
  channelsCoords: ChannelCoord[] | null;
  currentNeuralData: number[] | null;
}

// Brain outline component
function BrainOutline() {
  const lineRef = useRef<THREE.Line>(null);

  // Control points matrix [x, y, z]
  const controlPoints = [
    [-2.5,  1.5, -1.5], // Left back
    [-2.2,  1.8, -1.2], // Left back-top
    [-1.8,  2.0, -0.8], // Left top
    [-1.2,  2.2, -0.3], // Left upper-middle
    [-0.5,  2.3,  0.0], // Left middle
    [ 0.0,  2.3,  0.2], // Center upper
    [ 0.5,  2.3,  0.0], // Right middle
    [ 1.2,  2.2, -0.3], // Right upper-middle
    [ 1.8,  2.0, -0.8], // Right top
    [ 2.2,  1.8, -1.2], // Right back-top
    [ 2.5,  1.5, -1.5], // Right back
    [ 2.2,  1.0, -2.0], // Right bottom-back
    [ 1.8,  0.5, -2.3], // Right bottom
    [ 1.2,  0.2, -2.5], // Right lower
    [ 0.5,  0.0, -2.6], // Center bottom
    [ 0.0, -0.1, -2.6], // Bottom center
    [-0.5,  0.0, -2.6], // Left bottom
    [-1.2,  0.2, -2.5], // Left lower
    [-1.8,  0.5, -2.3], // Left bottom
    [-2.2,  1.0, -2.0], // Left bottom-back
    [-2.5,  1.5, -1.5], // Close loop: Left back
  ];

  // Create Catmull-Rom curve
  const curve = new THREE.CatmullRomCurve3(
    controlPoints.map(p => new THREE.Vector3(p[0], p[1], p[2]))
  );

  const points = curve.getPoints(100);
  const geometry = new THREE.BufferGeometry().setFromPoints(points);

  // Slow rotation animation
  useFrame((state, delta) => {
    if (lineRef.current) {
      lineRef.current.rotation.x += delta * 0.1;
      lineRef.current.rotation.y += delta * 0.2;
    }
  });

  return (
    <line>
      <bufferGeometry attach="geometry" {...geometry} />
      <lineBasicMaterial attach="material" color="#888888" />
    </line>
  );
}

// Axes helper component
function Axes() {
  return <axesHelper args={[5]} />;
}

export default function Brain3D({ channelsCoords, currentNeuralData }: Brain3DProps) {
  return (
    <div className="flex flex-col items-center">
      <h2 className="text-lg font-semibold text-primary-bg mb-4 text-center">Brain Top-Back View</h2>
      <div className="w-[400px] h-[400px] rounded-lg overflow-hidden border border-gray-700 shadow-2xl bg-white">
        <Canvas camera={{ position: [-6, 6, -6], fov: 60 }} style={{ background: '#ffffff' }}>
          <color attach="background" args={['#ffffff']} />
          <BrainOutline />
          <Axes />
          <OrbitControls />
        </Canvas>
      </div>
      <p className="text-xs text-gray-400 mt-2">Top-Back-Behind perspective</p>
    </div>
  );
}
