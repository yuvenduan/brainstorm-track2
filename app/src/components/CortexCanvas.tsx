'use client';

import { useRef, useEffect, useState } from 'react';

interface ScannerPosition {
  x: number;
  y: number;
}

interface BrainActivityData {
  position: { x: number; y: number };
  intensity: number;
}

interface CortexCanvasProps {
  brainData: BrainActivityData[] | null;
  gridSize: number;
  updateThrottle?: number; // milliseconds between updates
}

export default function CortexCanvas({ brainData, gridSize, updateThrottle = 100 }: CortexCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [brainImage, setBrainImage] = useState<HTMLImageElement | null>(null);
  const [scannerImage, setScannerImage] = useState<HTMLImageElement | null>(null);
  const [scannerPos, setScannerPos] = useState<ScannerPosition>({ x: 100, y: 100 });
  const [isScanning, setIsScanning] = useState(false);
  const [lastRenderTime, setLastRenderTime] = useState(0);

  // Load images
  useEffect(() => {
    const loadImages = async () => {
      try {
        // Load brain image
        const brainImg = new Image();
        brainImg.src = '/brain.png';
        brainImg.onload = () => {
          console.log('Brain image loaded:', brainImg.width, 'x', brainImg.height);
          setBrainImage(brainImg);
        };
        
        // Load scanner image
        const scannerImg = new Image();
        scannerImg.src = '/array.png';
        scannerImg.onload = () => {
          console.log('Scanner image loaded:', scannerImg.width, 'x', scannerImg.height);
          setScannerImage(scannerImg);
        };
      } catch (error) {
        console.error('Failed to load images:', error);
        // Try PNG fallback
        const brainImg = new Image();
        brainImg.src = '/brain.png';
        brainImg.onload = () => setBrainImage(brainImg);

        const scannerImg = new Image();
        scannerImg.src = '/array.png';
        scannerImg.onload = () => setScannerImage(scannerImg);
      }
    };

    loadImages();
  }, []);

  // Handle mouse movement for scanner positioning
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !isScanning) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setScannerPos({ x, y });
  };

  // Handle mouse click to toggle scanning
  const handleClick = () => {
    setIsScanning(!isScanning);
  };

  // Render the brain scan visualization
  const renderBrainScan = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    // Throttling - only render if enough time has passed
    const now = Date.now();
    if (now - lastRenderTime < updateThrottle && !isScanning) {
      return;
    }
    setLastRenderTime(now);

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Create vertical oval clipping path
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const ovalWidth = canvas.width * 0.35;
    const ovalHeight = canvas.height * 0.45;
    
    // Save context state
    ctx.save();
    
    // Create clipping mask with vertical oval
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, ovalWidth, ovalHeight, 0, 0, Math.PI * 2);
    ctx.clip();
    
    // Draw brain background with proper aspect ratio
    if (brainImage) {
      ctx.globalAlpha = 0.3;
      
      // Calculate proper scaling to fit within oval while preserving aspect ratio
      const brainAspectRatio = brainImage.width / brainImage.height;
      const ovalAspectRatio = ovalWidth / ovalHeight;
      
      let drawWidth, drawHeight, offsetX, offsetY;
      
      if (brainAspectRatio > ovalAspectRatio) {
        // Brain image is wider - fit to height
        drawHeight = ovalHeight * 2; // Full oval height
        drawWidth = drawHeight * brainAspectRatio;
        offsetX = centerX - drawWidth / 2;
        offsetY = centerY - ovalHeight;
      } else {
        // Brain image is taller - fit to width
        drawWidth = ovalWidth * 2; // Full oval width
        drawHeight = drawWidth / brainAspectRatio;
        offsetX = centerX - ovalWidth;
        offsetY = centerY - drawHeight / 2;
      }
      
      // Draw brain image within the clipped oval region
      ctx.drawImage(brainImage, offsetX, offsetY, drawWidth, drawHeight);
      ctx.globalAlpha = 1.0;
    } else {
      // Fallback: draw dim gray brain shape within oval
      ctx.globalAlpha = 0.3;
      ctx.fillStyle = '#4a5568';
      ctx.beginPath();
      ctx.ellipse(centerX, centerY, ovalWidth, ovalHeight, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1.0;
    }
    
    // Restore context state (removes clipping)
    ctx.restore();

    // Heatmap drawing logic is available but commented out
    // Uncomment this section to enable heatmap visualization
    /*
    if (brainData && brainData.length > 0) {
      brainData.forEach(data => {
        const { position, intensity } = data;
        const x = (position.x / gridSize) * canvas.width;
        const y = (position.y / gridSize) * canvas.height;
        
        // Simplified heatmap color calculation
        const clampedIntensity = Math.min(Math.max(intensity, 0), 1);
        const radius = 7;
        
        // Pre-calculated heatmap colors
        let r, g, b;
        if (clampedIntensity < 0.33) {
          r = 0;
          g = Math.floor(255 * (clampedIntensity * 3));
          b = Math.floor(255 * (1 - clampedIntensity * 3));
        } else if (clampedIntensity < 0.66) {
          r = Math.floor(255 * ((clampedIntensity - 0.33) * 3));
          g = 255;
          b = 0;
        } else {
          r = 255;
          g = Math.floor(255 * (1 - (clampedIntensity - 0.66) * 3));
          b = 0;
        }

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.6)`;
        ctx.fill();
      });
    }
    */

    // Draw scanner if loaded and scanning
    if (scannerImage && isScanning) {
      let scannerHeight;
      const scannerWidth = scannerHeight = 160;
      
      ctx.save();
      ctx.globalAlpha = 0.8;
      ctx.drawImage(
        scannerImage,
        scannerPos.x - scannerWidth / 2,
        scannerPos.y - scannerHeight / 2,
        scannerWidth,
        scannerHeight
      );
      ctx.restore();

      /*
      // Draw scanning indicator
      ctx.beginPath();
      ctx.arc(scannerPos.x, scannerPos.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = '#00ff00';
      ctx.fill();

       */
    }
  };

  // Animation loop - optimized to only render when data changes
  useEffect(() => {
    let animationFrameId: number;
    
    const animate = () => {
      renderBrainScan();
      // Only continue animation if scanning is active
      if (isScanning) {
        animationFrameId = requestAnimationFrame(animate);
      }
    };
    
    if (isScanning) {
      animate();
    } else {
      // Render once when not scanning
      renderBrainScan();
    }
    
    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [brainData, scannerPos, isScanning, brainImage, scannerImage]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={600}
        height={600}
        onMouseMove={handleMouseMove}
        onClick={handleClick}
        className="bg-gray-900 rounded-lg cursor-crosshair"
      />
      <div className="absolute bottom-4 left-4 bg-black bg-opacity-70 text-white px-3 py-2 rounded-lg text-sm">
        <div>Scanner: {isScanning ? 'ACTIVE' : 'INACTIVE'}</div>
        <div>Click to toggle scanning mode</div>
        <div>Move mouse to position scanner</div>
      </div>
    </div>
  );
}