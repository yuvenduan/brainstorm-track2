// Example usage in your page.tsx or parent component

import CortexCanvas from '@/components/CortexCanvas';

// Convert neural data to brain activity data format
const convertNeuralToBrainData = (neuralData: number[] | null, channelsCoords: any[] | null) => {
  if (!neuralData || !channelsCoords) return null;
  
  return neuralData.map((value, index) => ({
    position: {
      x: channelsCoords[index]?.x || 0,
      y: channelsCoords[index]?.y || 0
    },
    intensity: Math.abs(value) / 0.02 // Normalize to 0-1 range
  }));
};

// In your component render:
/*
<CortexCanvas 
  brainData={convertNeuralToBrainData(currentNeuralData, channelsCoords)}
  gridSize={gridSize}
/>
*/