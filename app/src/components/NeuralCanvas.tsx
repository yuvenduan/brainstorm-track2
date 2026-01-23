'use client';

import { useRef, useEffect } from 'react';

interface ChannelCoord {
  x: number;
  y: number;
}

interface NeuralCanvasProps {
  channelsCoords: ChannelCoord[] | null;
  gridSize: number;
  valueToColor: (value: number) => string;
  currentNeuralData: number[] | null;
}


export default function NeuralCanvas({ 
  channelsCoords,
  gridSize, 
  valueToColor,
  currentNeuralData
}: NeuralCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasSize = 300;
  const channelSize = (canvasSize*0.7) / gridSize;

  const getChannelPosition = (coord: ChannelCoord) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    
    const padding = 30;
    const plotSize = Math.min(canvas.width, canvas.height) - 2 * padding;
    
    // coords are 1-indexed, convert to 0-indexed
    const x = (coord.x - 1) / (gridSize - 1) * plotSize + padding;
    const y = (coord.y - 1) / (gridSize - 1) * plotSize + padding;
    
    return { x, y };
  };

  const renderNeuralData = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    console.log('Rendering canvas with data:', currentNeuralData, 'channels:', channelsCoords);
    if (!canvas || !ctx || !channelsCoords || !currentNeuralData) return;

    // Clear canvas
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw each channel
    console.log(`Drawing ${Math.min(channelsCoords.length, currentNeuralData.length)} channels`);
    for (let i = 0; i < Math.min(channelsCoords.length, currentNeuralData.length); i++) {
      const coord = channelsCoords[i];
      const value = currentNeuralData[i];
      const pos = getChannelPosition(coord);
      
      console.log(`Channel ${i}: coord=(${coord.x},${coord.y}), value=${value}, pos=(${pos.x},${pos.y})`);

      // Draw filled circle
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, channelSize / 2, 0, Math.PI * 2);
      ctx.fillStyle = valueToColor(value);
      ctx.fill();
    }
  };

  useEffect(() => {
    console.log('useEffect triggered - channelsCoords:', !!channelsCoords, 'currentNeuralData:', !!currentNeuralData);
    if (channelsCoords && currentNeuralData) {
      renderNeuralData();
    }
  }, [channelsCoords, currentNeuralData]);

  return (
    <canvas
      ref={canvasRef}
      width={canvasSize}
      height={canvasSize}
      className="bg-primary-bg rounded-lg border border-border"
    />
  );
}