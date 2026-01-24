'use client';

import { useState, useEffect, useRef } from 'react';
import NeuralCanvas from '@/components/NeuralCanvas';
import CortexCanvas from '@/components/CortexCanvas';
import StatusBar from '@/components/StatusBar';
import ConnectionControls from '@/components/ConnectionControls';

interface ChannelCoord {
  x: number;
  y: number;
}

interface WebSocketMessage {
  type: string;
  channels_coords?: [number, number][];
  grid_size?: number;
  neural_data?: number[][];
  start_time_s?: number;
  sample_count?: number;
  fs?: number;
}



const ws_url = "ws://localhost:8765"; //8766





// Generate magma colormap
const generateMagmaColormap = (): [number, number, number][] => {
  const magmaData = [
    [0.001462, 0.000466, 0.013866],
    [0.013708, 0.011771, 0.068667],
    [0.039608, 0.031090, 0.133515],
    [0.074257, 0.052017, 0.193510],
    [0.113094, 0.065492, 0.243537],
    [0.154901, 0.071327, 0.284065],
    [0.198177, 0.072245, 0.316356],
    [0.241397, 0.072699, 0.340836],
    [0.284124, 0.073417, 0.358296],
    [0.326438, 0.074167, 0.369846],
    [0.368567, 0.074621, 0.376400],
    [0.410791, 0.074866, 0.378497],
    [0.453187, 0.074686, 0.376427],
    [0.495784, 0.074295, 0.370369],
    [0.538516, 0.073859, 0.360437],
    [0.581246, 0.073480, 0.346753],
    [0.623796, 0.073307, 0.329512],
    [0.666022, 0.073590, 0.308947],
    [0.707797, 0.074578, 0.285380],
    [0.748980, 0.076556, 0.259246],
    [0.789417, 0.079868, 0.230962],
    [0.828991, 0.084937, 0.200963],
    [0.867534, 0.092252, 0.169642],
    [0.904837, 0.102306, 0.137338],
    [0.940621, 0.115594, 0.104286],
    [0.974449, 0.133635, 0.070619],
    [0.995560, 0.165380, 0.039886],
    [0.998085, 0.211843, 0.021563],
    [0.987053, 0.266188, 0.024335],
    [0.968443, 0.321898, 0.042144],
    [0.948683, 0.375586, 0.064264],
    [0.932067, 0.426710, 0.088087],
    [0.921248, 0.475767, 0.111534],
    [0.917482, 0.523424, 0.133798],
    [0.920858, 0.570213, 0.154815],
    [0.931674, 0.616411, 0.175091],
    [0.949545, 0.662198, 0.195563],
    [0.973381, 0.707719, 0.217587],
    [0.993248, 0.753418, 0.243755],
    [0.998364, 0.800551, 0.282327],
    [0.987622, 0.849251, 0.337977],
    [0.969680, 0.897560, 0.410320],
    [0.963855, 0.941167, 0.490000],
    [0.980600, 0.973500, 0.560100],
    [0.987053, 0.991438, 0.749504]
  ];

  const colormap: [number, number, number][] = [];
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (magmaData.length - 1);
    const idx = Math.floor(t);
    const frac = t - idx;

    if (idx >= magmaData.length - 1) {
      const c = magmaData[magmaData.length - 1];
      colormap.push([
        Math.round(c[0] * 255),
        Math.round(c[1] * 255),
        Math.round(c[2] * 255)
      ]);
    } else {
      const c1 = magmaData[idx];
      const c2 = magmaData[idx + 1];
      colormap.push([
        Math.round((c1[0] + frac * (c2[0] - c1[0])) * 255),
        Math.round((c1[1] + frac * (c2[1] - c1[1])) * 255),
        Math.round((c1[2] + frac * (c2[2] - c1[2])) * 255)
      ]);
    }
  }

  return colormap;
};

const MAGMA_COLORMAP = generateMagmaColormap();
const V_MIN = -0.02;
const V_MAX = 0.02;

const valueToColorIndex = (value: number): number => {
  const normalized = (value - V_MIN) / (V_MAX - V_MIN);
  const clamped = Math.max(0, Math.min(1, normalized));
  return Math.round(clamped * 255);
};

const valueToColor = (value: number): string => {
  const idx = valueToColorIndex(value);
  const [r, g, b] = MAGMA_COLORMAP[idx];
  return `rgb(${r}, ${g}, ${b})`;
};

export default function Home() {
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [statusText, setStatusText] = useState('Disconnected');
  const [channelsCoords, setChannelsCoords] = useState<ChannelCoord[] | null>(null);
  const [gridSize, setGridSize] = useState(32);
  const [currentTime, setCurrentTime] = useState(0.0);
  const [fps, setFps] = useState(0);
  const [channelCount, setChannelCount] = useState(0);
  const [currentNeuralData, setCurrentNeuralData] = useState<number[] | null>(null);

  const sampleBufferRef = useRef<number[][]>([]);
  const timeBufferRef = useRef<number[]>([]);
  const lastFrameTimeRef = useRef<number>(0);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef<number>(0);

  const TARGET_FPS = 30.0;
  const FRAME_INTERVAL = 1000.0 / TARGET_FPS;

  // Convert neural data to brain activity data for CortexCanvas
  const convertNeuralToBrainData = () => {
    if (!currentNeuralData || !channelsCoords) return null;

    return currentNeuralData.map((value, index) => ({
      position: {
        x: channelsCoords[index]?.x || 0,
        y: channelsCoords[index]?.y || 0
      },
      intensity: Math.min(Math.abs(value) / 0.01, 1) // Clamp to 0-1 range, less sensitive
    }));
  };
  const updateFpsCounter = () => {
    frameCountRef.current++;
    const now = performance.now();
    if (now - lastFpsUpdateRef.current >= 1000) {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      lastFpsUpdateRef.current = now;
    }
  };

  const emitFrame = () => {
    if (sampleBufferRef.current.length === 0) return;

    // Average all samples in buffer
    const nChannels = sampleBufferRef.current[0].length;
    const averagedData = new Array(nChannels).fill(0);

    for (let i = 0; i < sampleBufferRef.current.length; i++) {
      for (let j = 0; j < nChannels; j++) {
        averagedData[j] += sampleBufferRef.current[i][j];
      }
    }

    for (let j = 0; j < nChannels; j++) {
      averagedData[j] /= sampleBufferRef.current.length;
    }

    // Use the last timestamp
    const frameTime = timeBufferRef.current.length > 0
      ? timeBufferRef.current[timeBufferRef.current.length - 1]
      : 0.0;

    // Clear buffers
    sampleBufferRef.current = [];
    timeBufferRef.current = [];

    // Update state to trigger re-render
    console.log('Emitting frame with data:', averagedData);
    setCurrentNeuralData(averagedData);
    setCurrentTime(frameTime);
    updateFpsCounter();
  };

  const connect = (url: string) => {
    if (isConnected && ws) {
      ws.close();
      return;
    }

    setConnectionStatus('connecting');
    setStatusText('Connecting...');

    try {
      const websocket = new WebSocket(ws_url);

      websocket.onopen = () => {
        console.log('WebSocket connected');
        setWs(websocket);
        setIsConnected(true);
        setConnectionStatus('connected');
        setStatusText('Connected');
      };

      websocket.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data);

          if (data.type === 'init') {
            // Store channel coordinates and grid size
            if (data.channels_coords && Array.isArray(data.channels_coords)) {
              setChannelsCoords(data.channels_coords.map(coord => ({ x: coord[0], y: coord[1] })));
              setChannelCount(data.channels_coords.length);
            }
            if (data.grid_size) {
              setGridSize(data.grid_size);
            }

            console.log(`Initialized with ${data.channels_coords?.length || 0} channels, grid size ${data.grid_size}`);

            // Clear buffers on init
            sampleBufferRef.current = [];
            timeBufferRef.current = [];
            lastFrameTimeRef.current = performance.now();

          } else if (data.type === 'sample_batch') {
            // Accumulate samples from batch
            const neuralData = data.neural_data;
            const startTimeS = data.start_time_s || 0.0;
            const sampleCount = data.sample_count || neuralData?.length || 0;
            const fs = data.fs || 500.0;
            const dt = 1.0 / fs;

            // Add each sample to buffer
            if (neuralData) {
              for (let i = 0; i < sampleCount; i++) {
                const sampleTime = startTimeS + i * dt;
                if (i < neuralData.length) {
                  sampleBufferRef.current.push(neuralData[i]);
                  timeBufferRef.current.push(sampleTime);
                }
              }
            }

            // Check if it's time to emit a frame
            const currentTime = performance.now();
            if (currentTime - lastFrameTimeRef.current >= FRAME_INTERVAL) {
              emitFrame();
              lastFrameTimeRef.current = currentTime;
            }
          }
        } catch (err) {
          console.error('Error parsing message:', err);
        }
      };

      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('disconnected');
        setStatusText('Connection error');
      };

      websocket.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        setWs(null);
        setIsConnected(false);
        setConnectionStatus('disconnected');
        setStatusText(event.wasClean ? 'Disconnected' : 'Connection lost');
      };

    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setConnectionStatus('disconnected');
      setStatusText('Failed to connect');
    }
  };

  useEffect(()=>{
    document.addEventListener("keypress", function(event){
      if (event.key==="ArrowUp"){

      }
      if (event.key==="ArrowDown"){

      }
      if (event.key==="ArrowLeft"){

      }
      if (event.key==="ArrowRight"){

      }
    })
  })

  return (
    <main className="w-screen h-screen bg-white">
      <div className="flex flex-col items-center w-full h-full pt-6" style={{
        boxShadow: "0px 0px 70px 0px red inset"
      }}>
        {/*
        <header className="flex justify-between items-center pb-5 border-b border-border mb-6">
          <h1 className="text-xl font-semibold bg-gradient-to-r from-text-primary to-accent-primary bg-clip-text text-transparent">
            Neural Activity
          </h1>
          <StatusBar
            status={connectionStatus}
            statusText={statusText}
            currentTime={currentTime}
            fps={fps}
            channelCount={channelCount}
          />
        </header>
        */}

        <div className="flex ">
          {/* Brain Visualization */}
          <div className="flex flex-col items-center  p-6 rounded-xl">
            <h2 className="text-lg font-semibold text-primary-bg mb-4 text-center">Brain Scan Visualization</h2>
            <br/>
            <CortexCanvas
              brainData={convertNeuralToBrainData()}
              gridSize={gridSize}
              updateThrottle={200} // Update every 200ms when not scanning
            />
          </div>

          <div className="flex flex-col items-center bg-gray-200 p-6 rounded-lg">
            <div className="z-50">
              <div className="flex flex-col md:flex-row gap-6 justify-center items-center py-5">
                {/* Neural Canvas */}
                <div className="flex gap-5">
                  <NeuralCanvas
                      channelsCoords={channelsCoords}
                      gridSize={gridSize}
                      valueToColor={valueToColor}
                      currentNeuralData={currentNeuralData}
                  />
                  {/* right meter */}
                  <div className="flex gap-2 py-2">
                    <div className="w-4 rounded bg-gradient-to-b from-[#fcfdbf] via-[#feca8d] via-[#f1605d] via-[#b73779] via-[#721f81] to-[#2c115f] to-[#000004]"></div>
                    <div className="flex flex-col justify-between text-xs text-text-secondary">
                      <span>+0.02</span>
                      <span>0.0</span>
                      <span>-0.02</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <ConnectionControls
                isConnected={isConnected}
                onConnect={connect}
            />
          </div>
        </div>
      </div>
    </main>
  );
}

export {valueToColor};